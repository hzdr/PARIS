/*
 * This file is part of the ddafa reconstruction program.
 *
 * Copyright (C) 2016 Helmholtz-Zentrum Dresden-Rossendorf
 *
 * ddafa is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ddafa is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with ddafa. If not, see <http://www.gnu.org/licenses/>.
 *
 * Date: 02 December 2016
 * Authors: Jan Stephan <j.stephan@hzdr.de>
 */

#include <glados/cuda/coordinates.h>
#include <glados/cuda/launch.h>

#include "backend.h"

namespace ddafa
{
    namespace cuda
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

        } 

        namespace detail
        {
            auto call_weighting_kernel(float* dst, const float* src, std::uint32_t width, std::uint32_t height, std::size_t pitch,
                                       float h_min, float v_min, float d_sd, float l_px_row, float l_px_col, async_handle handle) -> void
            {
                glados::cuda::launch_async(handle, width, height,
                                            weighting_kernel,
                                            dst, src,
                                            width, height, pitch,
                                            h_min, v_min, d_sd, l_px_row, l_px_col);
            }
        }
    }
}
