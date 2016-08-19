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

#include <atomic>
#include <cmath>
#include <functional>
#include <future>
#include <map>
#include <thread>
#include <utility>

#include <boost/log/trivial.hpp>

#include <ddrf/cuda/coordinates.h>
#include <ddrf/cuda/launch.h>

#include "exception.h"
#include "metadata.h"
#include "weighting_stage.h"

namespace ddafa
{
    namespace kernel
    {
        __global__ void weight(float* output, const float* input,
                                std::size_t n_row, std::size_t n_col, std::size_t pitch,
                                float h_min, float v_min,
                                float d_sd,
                                float l_px_row, float l_px_col)
        {
            auto s = ddrf::cuda::coord_x();
            auto t = ddrf::cuda::coord_y();

            if((s < n_row) && (t < n_col))
            {
                auto input_row = reinterpret_cast<const float*>(reinterpret_cast<const char*>(input) + t * pitch);
                auto output_row = reinterpret_cast<float*>(reinterpret_cast<char*>(input) + t * pitch);

                // detector coordinates in mm
                auto h_s = (l_px_row / 2) + s * l_px_row + h_min;
                auto v_t = (l_px_col / 2) + t * l_px_col + v_min;

                // calculate weight
                auto w_st = d_sd * rsqrtf(powf(d_sd, 2) + powf(h_s, 2) + powf(v_t, 2));

                // write value
                output_row[s] = input_row[s] * w_st;
            }
        }
    }

    weighting_stage::weighting_stage(std::uint32_t n_row, std::uint32_t n_col,
                                        float l_px_row, float l_px_col,
                                        float delta_s, float delta_t,
                                        float d_so, float d_od) noexcept
    {
        h_min_ = delta_s * l_px_row - n_row * l_px_row / 2;
        v_min_ = delta_t * l_px_col - n_col * l_px_col / 2;
        d_sd_ = std::abs(d_so) + std::abs(d_od);
    }

    auto weighting_stage::run() -> void
    {
        auto devices = int{};
        auto err = cudaGetDeviceCount(&devices);
        if(err != cudaSuccess)
        {
            BOOST_LOG_TRIVIAL(fatal) << "weighting_stage::run() could not obtain CUDA devices: " << cudaGetErrorString(err);
            throw stage_runtime_error{"weighting_stage::run() failed to initialize"};
        }

        std::map<int, std::future<void>> futures;
        for(int i = 0; i < devices; ++i)
            futures[i] = std::async(std::launch::async, &weighting_stage::process, this, i);

        while(true)
        {

            auto proj = input_();
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

        try
        {
            for(auto&& fp : futures)
                fp.second.get();
        }
        catch(const stage_runtime_error& sre)
        {
            BOOST_LOG_TRIVIAL(fatal) << "weighting_stage::run() failed to execute: " << sre.what();
            throw stage_runtime_error{"weighting_stage::run() failed"};
        }

        output_(std::make_pair(nullptr, projection_metadata{0, 0, 0, 0.f, false, 0}));
        BOOST_LOG_TRIVIAL(info) << "Weighted all projections.";
    }

    auto weighting_stage::set_input_function(std::function<input_type(void)> input) noexcept -> void
    {
        input_ = input;
    }

    auto weighting_stage::set_output_function(std::function<void(output_type)> output) noexcept -> void
    {
        output_ = output;
    }

    auto weighting_stage::process(int device) -> void
    {
        auto err = cudaSetDevice(device);
        if(err != cudaSuccess)
        {
            BOOST_LOG_TRIVIAL(fatal) << "weighting_stage::process() could not set device: " << cudaGetErrorString(err);
            throw stage_runtime_error{"weighting_stage::process() failed to initialize"};
        }

        while(true)
        {
            while(input_map_.count(device) == 0)
                std::this_thread::yield();

            while(lock_.test_and_set(std::memory_order_acquire))
                std::this_thread::yield();

            auto& queue = input_map_.at(device);
            if(queue.empty())
            {
                lock_.clear(std::memory_order_release);
                continue;
            }
            auto proj = std::move(queue.front());
            queue.pop();

            lock_.clear(std::memory_order_release);

            if(!proj.second.valid)
                break;

            try
            {
                ddrf::cuda::launch(proj.second.width, proj.second.height,
                                    kernel::weight,
                                    proj.first.get(), static_cast<const float*>(proj.first.get()),
                                    n_row_, n_col_, proj.first.pitch(),
                                    h_min_, v_min_,
                                    d_sd_,
                                    l_px_row_, l_px_col_);
            }
            catch(const ddrf::cuda::bad_alloc& ba)
            {
                BOOST_LOG_TRIVIAL(fatal) << "weighting_stage::process() encountered bad_alloc while invoking kernel: " << ba.what();
                throw stage_runtime_error{"weighting_stage::process(): weighting kernel failed"};
            }
            catch(const ddrf::cuda::runtime_error& re)
            {
                BOOST_LOG_TRIVIAL(fatal) << "weighting_stage::process() encountered runtime_error while invoking kernel: " << re.what();
                throw stage_runtime_error{"weighting_stage::process(): weighting kernel failed"};
            }
            catch(const ddrf::cuda::invalid_argument& ia)
            {
                BOOST_LOG_TRIVIAL(fatal) << "weighting_stage::process() encountered invalid_argument while invoking kernel: " << ia.what();
                throw stage_runtime_error{"weighting_stage::process(): weighting kernel failed"};
            }

            output_(std::move(proj));
        }
    }
}
