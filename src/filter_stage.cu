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

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <utility>

#include <boost/log/trivial.hpp>

#include <cufft.h>

#include <ddrf/cuda/algorithm.h>
#include <ddrf/cuda/coordinates.h>
#include <ddrf/cuda/launch.h>
#include <ddrf/cuda/memory.h>
#include <ddrf/cuda/sync_policy.h>
#include <ddrf/cuda/utility.h>
#include <ddrf/cufft/plan.h>

#include "exception.h"
#include "geometry.h"
#include "filter_stage.h"

namespace ddafa
{
    namespace
    {
        __global__ void check(const float* in, std::uint32_t dim_x, std::uint32_t dim_y, std::size_t pitch)
        {
            auto x = ddrf::cuda::coord_x();
            auto y = ddrf::cuda::coord_y();

            if(x < dim_x && y < dim_y)
            {
                auto row = reinterpret_cast<const float*>(reinterpret_cast<const char*>(in) + y * pitch);

                if(x == 508 && y == 200)
                    printf("value = %f\n", row[x]);
            }
        }

        __global__ void filter_creation_kernel(float* __restrict__ r, const std::int32_t* __restrict__ j, std::uint32_t size, float tau)
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

        auto create_filter(std::uint32_t filter_size, float tau) -> ddrf::cuda::device_ptr<float>
        {
            /*
             * for a more detailed description see filter_creation_kernel
             */
            // create j on the host and fill it with values from -(filter_size_ - 2) / 2 to filter_size / 2
            auto h_j = ddrf::cuda::make_unique_pinned_host<std::int32_t>(filter_size);
            auto size = static_cast<std::int32_t>(filter_size);
            auto j = -(size - 2) / 2;
            std::iota(h_j.get(), h_j.get() + filter_size, j);
            BOOST_LOG_TRIVIAL(debug) << "Host filter creation succeeded";

            // create j on the device and copy j from the host to the device
            auto d_j = ddrf::cuda::make_unique_device<std::int32_t>(filter_size);
            ddrf::cuda::copy(ddrf::cuda::sync, d_j, h_j, filter_size);
            BOOST_LOG_TRIVIAL(debug) << "Copied filter from host to device";

            // create r on the device
            auto d_r  = ddrf::cuda::make_unique_device<float>(filter_size);

            // calculate the filter values
            ddrf::cuda::launch(filter_size, filter_creation_kernel, d_r.get(), static_cast<const std::int32_t*>(d_j.get()), filter_size, tau);
            BOOST_LOG_TRIVIAL(debug) << "Device filter creation succeeded";

            BOOST_LOG_TRIVIAL(debug) << "Filter creation complete";
            return d_r;
        }

        __global__ void k_creation_kernel(cufftComplex* __restrict__ data, std::uint32_t filter_size, float tau)
        {
            auto x = ddrf::cuda::coord_x();
            if(x < filter_size)
            {
                auto result = tau * fabsf(sqrtf(powf(data[x].x, 2.f) + powf(data[x].y, 2.f)));

                data[x].x = result;
                data[x].y = result;
            }
        }

        auto create_k(std::uint32_t size, float tau) -> ddrf::cuda::device_ptr<cufftComplex>
        {
            auto r = create_filter(size, tau);

            auto size_trans = size / 2 + 1;
            auto k = ddrf::cuda::make_unique_device<cufftComplex>(size_trans);

            auto n = static_cast<int>(size);

            auto plan = ddrf::cufft::plan<CUFFT_R2C>{n};
            plan.execute(r.get(), k.get());

            ddrf::cuda::launch(size_trans, k_creation_kernel, k.get(), size_trans, tau);

            return k;
        }

        __global__ void filter_application_kernel(cufftComplex* __restrict__ data, const cufftComplex* __restrict__ filter,
                                                    std::uint32_t filter_size, std::uint32_t data_height, std::size_t pitch)
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

        auto apply_filter(cufftComplex* in, const cufftComplex* k, std::uint32_t x, std::uint32_t y, std::size_t pitch, cudaStream_t stream) -> void
        {
            ddrf::cuda::launch_async(stream, x, y, filter_application_kernel, in, k, x, y, pitch);
        }

        __global__ void normalization_kernel(cufftReal* dst, std::size_t dst_pitch,
                                             const cufftReal* src, std::size_t src_pitch,
                                             std::uint32_t width, std::uint32_t height, std::uint32_t filter_size)
        {
            auto x = ddrf::cuda::coord_x();
            auto y = ddrf::cuda::coord_y();

            if((x < width) && (y < height))
            {
                auto dst_row = reinterpret_cast<cufftReal*>(reinterpret_cast<char*>(dst) + y * dst_pitch);
                auto src_row = reinterpret_cast<const cufftReal*>(reinterpret_cast<const char*>(src) + y * src_pitch);

                dst_row[x] = src_row[x] / filter_size;
            }
        }

        template <class In>
        auto normalize(In& in, std::uint32_t filter_size) -> void
        {
            ddrf::cuda::launch_async(in.stream, in.width, in.height,
                                        normalization_kernel,
                                        in.ptr.get(), in.ptr.pitch(),
                                        static_cast<const cufftReal*>(in.ptr.get()), in.ptr.pitch(),
                                        in.width, in.height, filter_size);
        }

        template <class In, class Out>
        auto expand(const In& in, Out& out, std::uint32_t x, std::uint32_t y) -> void
        {
            // reset expanded projection
            ddrf::cuda::fill(ddrf::cuda::async, out, 0, in.stream, x, y);

            // copy original projection to expanded projection
            ddrf::cuda::copy(ddrf::cuda::async, out, in.ptr, in.stream, in.width, in.height);
        }

        template <class In, class Out>
        auto shrink(const In& in, Out& out) -> void
        {
            ddrf::cuda::copy(ddrf::cuda::async, out.ptr, in, out.stream, out.width, out.height);
        }

        template <class In, class Out, class Plan>
        auto transform(In* in, Out* out, Plan& plan, cudaStream_t stream) -> void
        {
            plan.set_stream(stream);
            plan.execute(in, out);
        }

    }

    filter_stage::filter_stage(int device) noexcept
    : device_{device}
    {}

    auto filter_stage::assign_task(task t) noexcept -> void
    {
        filter_size_ = static_cast<std::uint32_t>(2 * std::pow(2, std::ceil(std::log2(t.det_geo.n_row))));
        n_col_ = t.det_geo.n_col;
        tau_ = t.det_geo.l_px_row;
    }

    auto filter_stage::run() -> void
    {
        auto sre = stage_runtime_error{"filter_stage::run() failed"};

        try
        {
            ddrf::cuda::set_device(device_);

            // create filter
            auto k = create_k(filter_size_, tau_);

            // dimensionality of the FFT - 1D in this case
            constexpr auto rank = 1;

            // size of the FFT for each dimension
            auto n = static_cast<int>(filter_size_);

            // we are executing a batched FFT -> set batch size
            auto batch = static_cast<int>(n_col_);

            // allocate memory for expanded projection (projection width -> filter_size_)
            auto p_exp = ddrf::cuda::make_unique_device<float>(filter_size_, n_col_);

            // allocate memory for transformed projection (filter_size_ -> size_trans)
            auto size_trans = filter_size_ / 2 + 1;
            auto p_trans = ddrf::cuda::make_unique_device<cufftComplex>(size_trans, n_col_);

            // calculate the distance between the first elements of two successive lines (needed for cuFFT)
            auto p_exp_dist = static_cast<int>(p_exp.pitch() / sizeof(float));
            auto p_trans_dist = static_cast<int>(p_trans.pitch() / sizeof(cufftComplex));

            // set the distance between two successive elements
            constexpr auto p_exp_stride = 1;
            constexpr auto p_trans_stride = 1;

            // set storage dimensions of data in memory
            auto p_exp_nembed = p_exp_dist;
            auto p_trans_nembed = p_trans_dist;

            // create plans for forward and inverse FFT
            auto forward = ddrf::cufft::plan<CUFFT_R2C>{rank, &n,
                                                        &p_exp_nembed, p_exp_stride, p_exp_dist,
                                                        &p_trans_nembed, p_trans_stride, p_trans_dist,
                                                        batch};

            auto inverse = ddrf::cufft::plan<CUFFT_C2R>{rank, &n,
                                                        &p_trans_nembed, p_trans_stride, p_trans_dist,
                                                        &p_exp_nembed, p_exp_stride, p_exp_dist,
                                                        batch};

            ddrf::cuda::synchronize_stream();

            BOOST_LOG_TRIVIAL(debug) << "Filter setup on device #" << device_ << " completed.";

            while(true)
            {
                auto p = input_();
                if(!p.valid)
                    break;

                ddrf::cuda::launch(p.width, p.height, check, static_cast<const float*>(p.ptr.get()), p.width, p.height, p.ptr.pitch());

                // expand and transform the projection
                expand(p, p_exp, filter_size_, n_col_);
                transform(p_exp.get(), p_trans.get(), forward, p.stream);

                // apply the filter to the transformed projection
                apply_filter(p_trans.get(), k.get(), filter_size_, n_col_, p_trans.pitch(), p.stream);

                // inverse transformation
                transform(p_trans.get(), p_exp.get(), inverse, p.stream);

                // shrink to original size and normalize
                shrink(p_exp, p);
                normalize(p, filter_size_);

                // done
                ddrf::cuda::synchronize_stream(p.stream);
                output_(std::move(p));
            }

            output_(output_type{});
            BOOST_LOG_TRIVIAL(info) << "All projections have been filtered.";
        }
        catch(const ddrf::cuda::bad_alloc& ba)
        {
            BOOST_LOG_TRIVIAL(fatal) << "filter_stage::run() encountered a bad_alloc: " << ba.what();
            throw sre;
        }
        catch(const ddrf::cuda::invalid_argument& ia)
        {
            BOOST_LOG_TRIVIAL(fatal) << "filter_stage::run() passed an invalid argument to the CUDA runtime: " << ia.what();
            throw sre;
        }
        catch(const ddrf::cuda::runtime_error& re)
        {
            BOOST_LOG_TRIVIAL(fatal) << "filter_stage::run() caused a CUDA runtime error: " << re.what();
            throw sre;
        }
        catch(const ddrf::cufft::bad_alloc& ba)
        {
            BOOST_LOG_TRIVIAL(fatal) << "filter_stage::run() encountered a bad allocation in cuFFT: " << ba.what();
            throw sre;
        }
        catch(const ddrf::cufft::invalid_argument& ia)
        {
            BOOST_LOG_TRIVIAL(fatal) << "filter_stage::run() passed an invalid argument to cuFFT: " << ia.what();
            throw sre;
        }
        catch(const ddrf::cufft::runtime_error& re)
        {
            BOOST_LOG_TRIVIAL(fatal) << "filter_stage::run() encountered a cuFFT runtime error: " << re.what();
            throw sre;
        }
    }

    auto filter_stage::set_input_function(std::function<input_type(void)> input) noexcept -> void
    {
        input_ = input;
    }

    auto filter_stage::set_output_function(std::function<void(output_type)> output) noexcept -> void
    {
        output_ = output;
    }
}
