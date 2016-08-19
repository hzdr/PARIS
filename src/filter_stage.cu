/*
 * This file is part of the ddafa reconstruction program.
 *
 * Copyright (C) 2016 Helmholtz-Zentrum Dresden-Rossendorf
 *
 * Licensed under the EUPL, Version 1.1 or - as soon they will be approved by
 * the European Commission - subsequent version of the EUPL (the "Licence");
 * You may not use this work except in compliance with the Licence.
 * You may obtain a copy of the Licence at:
 *
 * http://ec.europa.eu/idabc/eupl
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the Licence is distributed on an "AS IS" basis,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the Licence for the specific language governing permissions and
 * limitations under the Licence.
 *
 * Date: 18 August 2016
 * Authors: Jan Stephan
 */

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <future>
#include <map>
#include <thread>
#include <utility>
#include <vector>

#include <boost/log/trivial.hpp>

#include <cufft.h>

#include <ddrf/cuda/algorithm.h>
#include <ddrf/cuda/coordinates.h>
#include <ddrf/cuda/launch.h>
#include <ddrf/cuda/memory.h>
#include <ddrf/cuda/sync_policy.h>
#include <ddrf/cufft/plan.h>

#include "exception.h"
#include "filter_stage.h"

namespace ddafa
{
    namespace kernel
    {
        __global__ void create_filter(float* __restrict__ r, const std::int32_t* __restrict__ j, std::size_t size, float tau)
        {
            auto x = ddrf::cuda::coord_x();

            /*
             * r(j) with j = [ -(filter_length - 2) / 2, ..., 0, ..., filter_length / 2 ]
             * tau = horizontal pixel distance
             *
             *          1/8 * 1/(tau^2)                     for j = 0
             * r(j) = { 0                                   for even j
             *          -(1 / (2 * j^2 * pi^2 * tau^2))     for odd j
             */
            if(x < size)
            {
                if(j[x] == 0) // does j = 0?
                    r[x] = (1.f / 8.f) * (1.f / powf(tau, 2.f));
                else // j != 0
                {
                    if(j[x] % 2 == 0) // is j even?
                        r[x] = 0.f;
                    else // j is odd
                        r[x] = -(1.f / (2.f * powf(j[x], 2.f) * powf(M_PI, 2.f) * powf(tau, 2.f)));
                }
            }
        }

        __global__ void create_k(cufftComplex* __restrict__ data, std::size_t filter_size, float tau)
        {
            auto x = ddrf::cuda::coord_x();
            if(x < filter_size)
            {
                auto result = tau * fabsf(sqrtf(powf(data[x].x, 2.f) + powf(data[x].y, 2.f)));
                data[x].x = result;
                data[x].y = result;
            }
        }

        __global__ void apply_filter(cufftComplex* __restrict__ data, const cufftComplex* __restrict__ filter,
                                        std::size_t filter_size, std::size_t data_height, std::size_t pitch)
        {
            auto x = ddrf::cuda::coord_x();
            auto y = ddrf::cuda::coord_y();

            if((x < filter_size) && (y < data_height))
            {
                auto row = reinterpret_cast<cufftComplex*>(reinterpret_cast<char*>(data) + y * pitch);

                row[x].x *= filter[x].x;
                row[x].y *= filter[x].y;
            }
        }
    }

    filter_stage::filter_stage(std::uint32_t n_row, std::uint32_t n_col, float l_px_row)
    : filter_length_{2 * std::pow(2, std::ceil(std::log2(n_row)))}
    , n_col_{n_col}
    , tau_{l_px_row}
    {
        auto err = cudaGetDeviceCount(&devices_);
        if(err != cudaSuccess)
        {
            BOOST_LOG_TRIVIAL(fatal) << "filter_stage::filter_stage() could not obtain CUDA devices: " << cudaGetErrorString(err);
            throw stage_construction_error{"filter_stage::filter_stage() failed"};
        }

        auto filter_futures = std::vector<std::future>{};
        for(auto i = 0; i < devices_; ++i)
            filter_futures.emplace_back(std::async(std::launch::async, &filter_stage::create_filter, this, i));

        try
        {
            for(auto&& f : filter_futures)
                f.get();
        }
        catch(const stage_construction_error& sce)
        {
            BOOST_LOG_TRIVIAL(fatal) << "filter_stage::filter_stage() could not create filters: " << sce.what();
            throw stage_construction_error{"filter_stage::filter_stage() failed"};
        }
    }

    auto filter_stage::run() -> void
    {
        std::map<int, std::future<void>> futures;
        for(int i = 0; i < devices_; ++i)
            futures[i] = std::async(std::launch::async, &filter_stage::process, this, i);

        while(true)
        {
            auto proj = input_();
            safe_push(std::move(proj));
        }

        try
        {
            for(auto&& fp : futures)
                fp.second.get();
        }
        catch(const stage_runtime_error& sre)
        {
            BOOST_LOG_TRIVIAL(fatal) << "filter_stage::run() failed to execute: " << sre.what();
            throw stage_runtime_error{"filter_stage::run() failed"};
        }

        output_(output_type());
        BOOST_LOG_TRIVIAL(info) << "All projections have been filtered.";
    }

    auto filter_stage::set_input_function(std::function<input_type(void)> input) noexcept -> void
    {
        input_ = input;
    }

    auto filter_stage::set_output_function(std::function<void(output_type)> output) noexcept -> void
    {
        output_ = output;
    }

    auto filter_stage::safe_push(input_type proj) -> void
    {
        while(lock_.test_and_set(std::memory_order_acquire))
            std::this_thread::yield();

        if(proj.second.valid)
            input_map_[proj.second.device].push(std::move(proj));
        else
        {
            for(auto i = 0; i < devices; ++i)
                input_map_[i].push(input_type());
        }

        lock_.clear(std::memory_order_release);
    }

    auto filter_stage::safe_pop(int device) -> input_type
    {
        while(input_map_.count(device) == 0)
            std::this_thread::yield();

        while(lock_.test_and_set(std::memory_order_acquire))
            std::this_thread::yield();

        auto& queue = input_map_[device];
        if(queue.empty())
        {
            lock_.clear(std::memory_order_release);
            continue;
        }
        auto proj = std::move(queue.front());
        queue.pop();

        lock_.clear(std::memory_order_release);

        return proj;
    }

    auto filter_stage::create_filter(int device) -> void
    {
        auto err = cudaSetDevice(device);
        if(err != cudaSuccess)
        {
            BOOST_LOG_TRIVIAL(fatal) << "filter_stage::create_filter() could not set CUDA device: " << cudaGetErrorString(err);
            throw stage_construction_error{"filter_stage::create_filter() failed"};
        }

        try
        {
            /*
             * for a more detailed description see kernel::create_filter()
             */
            // create j on the host and fill it with values from -(filter_size_ - 2) / 2 to filter_size / 2
            auto h_j = ddrf::cuda::make_unique_host<std::int32_t>(filter_size_);
            auto size = static_cast<std::int32_t>(filter_size_);
            auto j = (size - 2) / 2;
            std::iota(h_j.get(), h_j.get() + filter_size_, j);

            // create j on the device and copy j from the host to the device
            auto d_j = ddrf::cuda::make_unique_device<float>(filter_size_);
            ddrf::cuda::copy(ddrf::cuda::async, d_j, h_j, filter_size_);

            // create r on the device and calculate the filter values
            auto d_r = ddrf::cuda::make_unique_device<float>(filter_size_);
            ddrf::cuda::launch(filter_size_, kernel::create_filter, d_r.get(), static_cast<const std::int32_t*>(d_j.get()), filter_size_, tau_);

            // move to filter container
            rs_[device] = std::move(d_r);
        }
        catch(const ddrf::cuda::bad_alloc& ba)
        {
            BOOST_LOG_TRIVIAL(fatal) << "filter_stage::create_filter() encountered bad_alloc: " << ba.what();
            throw stage_construction_error{"filter_stage::create_filter() failed"};
        }
        catch(const ddrf::cuda::runtime_error& re)
        {
            BOOST_LOG_TRIVIAL(fatal) << "filter_stage::create_filter() encountered runtime_error: " << re.what();
            throw stage_construction_error{"filter_stage::create_filter() failed"};
        }
        catch(const ddrf::cuda::invalid_argument& ia)
        {
            BOOST_LOG_TRIVIAL(fatal) << "filter_stage::create_filter() encountered invalid_argument: " << ia.what();
            throw stage_construction_error{"filter_stage::create_filter() failed"};
        }
    }

    auto filter_stage::process(int device) -> void
    {
        auto err = cudaSetDevice(device);
        if(err != cudaSuccess)
        {
            BOOST_LOG_TRIVIAL(fatal) << "filter_stage::process() could not set CUDA device: " << cudaGetErrorString(err);
            throw stage_runtime_error{"filter_stage::process() failed"};
        }

        // allocate memory for projection conversion and transformation
        auto converted_proj = ddrf::cuda::pitched_device_ptr<float>{nullptr};
        auto transformed_proj = ddrf::cuda::pitched_device_ptr<cufftComplex>{nullptr};

        auto transformed_filter = ddrf::cuda::device_ptr<cufftComplex>{nullptr};
        auto transformed_filter_size = filter_size_ / 2 + 1;
        try
        {
            converted_proj = ddrf::cuda::make_unique_device<float>(filter_size_, n_col_);
            transformed_proj = ddrf::cuda::make_unique_device<cufftComplex(transformed_filter_size, n_col_);
            transformed_filter_ = ddrf::cuda::make_unique_device<cufftComplex>(transformed_filter_size);

            ddrf::cuda::fill(ddrf::cuda::async, converted_proj, 0, filter_size_, n_col_);
        }
        catch(const ddrf::cuda::bad_alloc& ba)
        {
            BOOST_LOG_TRIVIAL(fatal) << "filter_stage::process() encountered bad_alloc: " << ba.what();
            throw stage_runtime_error{"filter_stage::process() failed"};
        }
        catch(const ddrf::cuda::invalid_argument& ia)
        {
            BOOST_LOG_TRIVIAL(fatal) << "filter_stage::process() encountered invalid_argument: " << ia.what();
            throw stage_runtime_error{"filter_stage::process() failed"};
        }

        // set up cuFFT
        auto converted_proj_plan = ddrf::cufft::plan<CUFFT_R2C>{};
        auto filter_plan = ddrf::cufft::plan<CUFFT_R2C>{};
        auto inverse_plan = ddrf::cufft::plan<CUFFT_C2R>{};

        auto proj_n = int{filter_size_};
        auto proj_dist = int{converted_proj.pitch() / sizeof(float)};
        auto proj_nembed = proj_dist;

        auto trans_dist = int{transformed_proj.pitch() / sizeof(cufftComplex)};
        auto trans_nembed = trans_dist;

        try
        {
            converted_proj_plan = ddrf::cufft::plan<CUFFT_R2C>{1, &proj_n, &proj_nembed, 1, proj_dist, trans_nembed, 1, trans_dist, n_col_};
            filter_plan = ddrf::cufft::plan<CUFFT_R2C>{proj_n};
            inverse_plan = ddrf::cufft::plan<CUFFT_C2R>{1, &proj_n, &trans_nembed, 1, trans_dist, &proj_nembed, 1, proj_dist, n_col_};
        }
        catch(const ddrf::cufft::bad_alloc& ba)
        {
            BOOST_LOG_TRIVIAL(fatal) << "filter_stage::process() encountered a bad allocation in cuFFT: " << ba.what();
            throw stage_runtime_error{"filter_stage::process() failed"};
        }

        while(true)
        {
            auto proj = safe_pop(device);
            if(!proj.second.valid)
                break;

            try
            {
                // copy projection to larger projection which has a width of 2^x
                ddrf::cuda::copy(ddrf::cuda::async, converted_proj, proj.first, proj.second.width, proj.second.height);

                // execute the FFT for the projection and the filter
                converted_proj_plan.execute(converted_proj.get(), transformed_proj.get());
                filter_plan.execute(rs_[device].get(), transformed_filter.get());

                // create K
                ddrf::cuda::launch(transformed_filter_size, create_k, transformed_filter.get(), transformed_filter_size, tau_);

                // apply the transformed filter to the transformed projection
                ddrf::cuda::launch(transformed_filter_size, n_col_,
                                    apply_filter,
                                    transformed_proj.get(), transformed_filter.get(), transformed_filter_size, n_col_, transformed_proj.pitch());

                // run inverse FFT on the transformed projection
                inverse_plan.execute(transformed_proj.get(), converted_proj.get());

                // copy back to original projection dimensions
                ddrf::cuda::copy(ddrf::cuda::async, proj.first, converted_proj, proj.second.width, proj.second.height);

                output_(std::move(proj));
            }
            catch(const ddrf::cufft::bad_alloc& ba)
            {
                BOOST_LOG_TRIVIAL(fatal) << "filter_stage::process() encountered a bad allocation in cuFFT: " << ba.what();
                throw stage_runtime_error{"filter_stage::process() failed"};
            }
            catch(const ddrf::cufft::invalid_argument& ia)
            {
                BOOST_LOG_TRIVIAL(fatal) << "filter_stage::process() passed an invalid argument to cuFFT: " << ia.what();
                throw stage_runtime_error{"filter_stage::process() failed"};
            }
            catch(const ddrf::cufft::runtime_error& re)
            {
                BOOST_LOG_TRIVIAL(fatal) << "filter_stage::process() encountered a cuFFT runtime error: " << re.what();
                throw stage_runtime_error{"filter_stage::process() failed"};
            }
            catch(const ddrf::cuda::invalid_argument& ia)
            {
                BOOST_LOG_TRIVIAL(fatal) << "filter_stage::process() passed an invalid argument to the CUDA runtime: " << ia.what();
                throw stage_runtime_error{"filter_stage::process() failed"};
            }
        }
    }
}
