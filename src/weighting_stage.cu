/*
 * This file is part of the PARIS reconstruction program.
 *
 * Copyright (C) 2016 Helmholtz-Zentrum Dresden-Rossendorf
 *
 * PARIS is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PARIS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PARIS. If not, see <http://www.gnu.org/licenses/>.
 *
 * Date: 18 August 2016
 * Authors: Jan Stephan <j.stephan@hzdr.de>
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <utility>

#include <boost/log/trivial.hpp>

#include <glados/cuda/coordinates.h>
#include <glados/cuda/exception.h>
#include <glados/cuda/launch.h>
#include <glados/cuda/utility.h>

#include "exception.h"
#include "geometry.h"
#include "projection.h"
#include "weighting_stage.h"

namespace paris
{
    namespace
    {
        __global__ void weighting_kernel(float* output, const float* input,
                                std::uint32_t n_row, std::uint32_t n_col, std::size_t pitch,
                                float h_min, float v_min,
                                float d_sd,
                                float l_px_row, float l_px_col)
        {
            auto s = glados::cuda::coord_x();
            auto t = glados::cuda::coord_y();

            if((s < n_row) && (t < n_col))
            {
                auto input_row = reinterpret_cast<const float*>(reinterpret_cast<const char*>(input) + t * pitch);
                auto output_row = reinterpret_cast<float*>(reinterpret_cast<char*>(output) + t * pitch);

                // enable parallel global memory fetch while calculating
                auto val = input_row[s];

                // detector coordinates in mm
                auto h_s = (l_px_row / 2) + s * l_px_row + h_min;
                auto v_t = (l_px_col / 2) + t * l_px_col + v_min;

                // calculate weight
                auto w_st = d_sd * rsqrtf(powf(d_sd, 2) + powf(h_s, 2) + powf(v_t, 2));

                // write value
                output_row[s] = val * w_st;
            }
        }

        template <class In>
        auto weight(In& p, float h_min, float v_min, float d_sd, float l_px_row, float l_px_col) -> void
        {
            glados::cuda::launch_async(p.stream, p.width, p.height,
                                weighting_kernel,
                                p.ptr.get(), static_cast<const float*>(p.ptr.get()),
                                p.width, p.height, p.ptr.pitch(),
                                h_min, v_min, d_sd, l_px_row, l_px_col);
        }
    }

    weighting_stage::weighting_stage(int device) noexcept
    : device_{device}
    {}

    auto weighting_stage::assign_task(task t) noexcept -> void
    {
        det_geo_ = t.det_geo;

        auto n_row_f = static_cast<float>(det_geo_.n_row);
        auto n_col_f = static_cast<float>(det_geo_.n_col);
        h_min_ = det_geo_.delta_s * det_geo_.l_px_row - n_row_f * det_geo_.l_px_row / 2;
        v_min_ = det_geo_.delta_t * det_geo_.l_px_col - n_col_f * det_geo_.l_px_col / 2;
        d_sd_ = std::abs(det_geo_.d_so) + std::abs(det_geo_.d_od);
    }

    auto weighting_stage::run() const -> void
    {
        auto sre = stage_runtime_error{"weighting_stage::run() failed"};

        try
        {
            glados::cuda::set_device(device_);
            while(true)
            {
                auto p = input_();
                if(!p.valid)
                    break;

                // weight the projection
                weight(p, h_min_, v_min_, d_sd_, det_geo_.l_px_row, det_geo_.l_px_col);

                // done
                glados::cuda::synchronize_stream(p.stream);
                output_(std::move(p));
            }

            output_(output_type{});
            BOOST_LOG_TRIVIAL(info) << "Weighted all projections.";
        }
        catch(const glados::cuda::bad_alloc& ba)
        {
            BOOST_LOG_TRIVIAL(fatal) << "weighting_stage::run() encountered a bad_alloc: " << ba.what();
            throw sre;
        }
        catch(const glados::cuda::invalid_argument& ia)
        {
            BOOST_LOG_TRIVIAL(fatal) << "weighting_stage::run() passed an invalid argument to the CUDA runtime: " << ia.what();
            throw sre;
        }
        catch(const glados::cuda::runtime_error& re)
        {
            BOOST_LOG_TRIVIAL(fatal) << "weighting_stage::run() caused a CUDA runtime error: " << re.what();
            throw sre;
        }
    }

    auto weighting_stage::set_input_function(std::function<input_type(void)> input) noexcept -> void
    {
        input_ = input;
    }

    auto weighting_stage::set_output_function(std::function<void(output_type)> output) noexcept -> void
    {
        output_ = output;
    }
}
