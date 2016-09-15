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
#include <functional>
#include <future>
#include <mutex>
#include <thread>
#include <utility>

#include <boost/log/trivial.hpp>

#include <ddrf/cuda/coordinates.h>
#include <ddrf/cuda/exception.h>
#include <ddrf/cuda/launch.h>
#include <ddrf/cuda/utility.h>

#include "exception.h"
#include "projection.h"
#include "weighting_stage.h"

namespace ddafa
{
    namespace
    {
        std::mutex mutex__;

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
                auto output_row = reinterpret_cast<float*>(reinterpret_cast<char*>(output) + t * pitch);

                // detector coordinates in mm
                auto h_s = (l_px_row / 2) + s * l_px_row + h_min;
                auto v_t = (l_px_col / 2) + t * l_px_col + v_min;

                // calculate weight
                auto w_st = d_sd * rsqrtf(powf(d_sd, 2) + powf(h_s, 2) + powf(v_t, 2));

                // write value
                auto val = input_row[s] * w_st;
                output_row[s] = val;
            }
        }
    }

    weighting_stage::weighting_stage(std::uint32_t n_row, std::uint32_t n_col,
                                     float l_px_row, float l_px_col,
                                     float delta_s, float delta_t,
                                     float d_so, float d_od)
    : l_px_row_{l_px_row}, l_px_col_{l_px_col}
    {
        auto sce = stage_construction_error{"weighting_stage::weighting_stage() failed"};

        try
        {
            auto n_row_f = static_cast<float>(n_row);
            auto n_col_f = static_cast<float>(n_col);
            h_min_ = delta_s * l_px_row - n_row_f * l_px_row / 2;
            v_min_ = delta_t * l_px_col - n_col_f * l_px_col / 2;
            d_sd_ = std::abs(d_so) + std::abs(d_od);

            devices_ = ddrf::cuda::get_device_count();

            auto d = static_cast<iv_size_type>(devices_);
            input_vec_ = decltype(input_vec_){d};
        }
        catch(const ddrf::cuda::bad_alloc& ba)
        {
            BOOST_LOG_TRIVIAL(fatal) << "weighting_stage::weighting_stage() encountered a bad_alloc: " << ba.what();
            throw sce;
        }
        catch(const ddrf::cuda::invalid_argument& ia)
        {
            BOOST_LOG_TRIVIAL(fatal) << "weighting_stage::weighting_stage() passed an invalid argument to the CUDA runtime: " << ia.what();
            throw sce;
        }
        catch(const ddrf::cuda::runtime_error& re)
        {
            BOOST_LOG_TRIVIAL(fatal) << "weighting_stage::weighting_stage() caused a CUDA runtime error: " << re.what();
            throw sce;
        }
    }

    auto weighting_stage::run() -> void
    {
        auto futures = std::vector<std::future<void>>{};
        for(auto i = 0; i < devices_; ++i)
            futures.emplace_back(std::async(std::launch::async, &weighting_stage::process, this, i));

        auto l = std::unique_lock<std::mutex>{mutex__, std::defer_lock};

        while(true)
        {
            auto proj = input_();
            auto valid = proj.valid;

            l.lock();

            if(valid)
            {
                auto d = static_cast<iv_size_type>(proj.device);
                input_vec_.at(d).push(std::move(proj));
            }

            else
            {
                for(auto i = 0; i < devices_; ++i)
                {
                    auto d = static_cast<iv_size_type>(i);
                    input_vec_.at(d).push(input_type{});
                }
            }

            l.unlock();

            if(!valid)
                break;
        }

        for(auto&& f : futures)
            f.get();

        output_(output_type{});
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
        auto sre = stage_runtime_error{"weighting_stage::process() failed"};

        try
        {
            ddrf::cuda::set_device(device);

            auto l = std::unique_lock<std::mutex>{mutex__, std::defer_lock};

            while(input_vec_.empty())
                std::this_thread::yield();

            auto d = static_cast<iv_size_type>(device);
            auto&& queue = input_vec_.at(d);

            while(true)
            {
                l.lock();

                if(queue.empty())
                {
                    l.unlock();
                    continue;
                }

                auto proj = std::move(queue.front());
                queue.pop();

                l.unlock();

                if(!proj.valid)
                    break;

                ddrf::cuda::launch_async(proj.stream, proj.width, proj.height,
                                    weight,
                                    proj.ptr.get(), static_cast<const float*>(proj.ptr.get()),
                                    proj.width, proj.height, proj.ptr.pitch(),
                                    h_min_, v_min_,
                                    d_sd_,
                                    l_px_row_, l_px_col_);

                ddrf::cuda::synchronize_stream(proj.stream);

                output_(std::move(proj));
            }
        }
        catch(const ddrf::cuda::bad_alloc& ba)
        {
            BOOST_LOG_TRIVIAL(fatal) << "weighting_stage::process() encountered a bad_alloc: " << ba.what();
            throw sre;
        }
        catch(const ddrf::cuda::invalid_argument& ia)
        {
            BOOST_LOG_TRIVIAL(fatal) << "weighting_stage::process() passed an invalid argument to the CUDA runtime: " << ia.what();
            throw sre;
        }
        catch(const ddrf::cuda::runtime_error& re)
        {
            BOOST_LOG_TRIVIAL(fatal) << "weighting_stage::process() caused a CUDA runtime error: " << re.what();
            throw sre;
        }
    }
}
