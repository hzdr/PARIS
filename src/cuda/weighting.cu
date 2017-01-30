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
 * Date: 02 December 2016
 * Authors: Jan Stephan <j.stephan@hzdr.de>
 */

#include <glados/cuda/coordinates.h>
#include <glados/cuda/launch.h>
#include <glados/cuda/utility.h>

#include "backend.h"

namespace paris
{
    namespace cuda
    {
        namespace
        {
            __global__ void weighting_kernel(float* p, std::uint32_t dim_x, std::uint32_t dim_y, std::size_t pitch,
                                             float h_min, float v_min, float d_sd, float l_px_row, float l_px_col)
            {
                auto s = glados::cuda::coord_x();
                auto t = glados::cuda::coord_y();

                if((s < dim_x) && (t < dim_y))
                {
                    auto row = reinterpret_cast<float*>(reinterpret_cast<char*>(p) + t * pitch);

                    // enable parallel global memory fetch while calculating
                    const auto val = row[s];

                    // detector coordinates in mm
                    const auto h_s = (l_px_row / 2.f) + s * l_px_row + h_min;
                    const auto v_t = (l_px_col / 2.f) + t * l_px_col + v_min;

                    // calculate weight
                    const auto w_st = d_sd * rsqrtf(powf(d_sd, 2) + powf(h_s, 2) + powf(v_t, 2));

                    // write value
                    row[s] = val * w_st;
                }
            }

        } 

        auto weight(projection_device_type& p, float h_min, float v_min, float d_sd, float l_px_row, float l_px_col)
            -> void
        {
            thread_local static auto s = cuda_stream{};
            
            glados::cuda::launch_async(s.stream, p.dim_x, p.dim_y,
                                        weighting_kernel,
                                        p.buf.get(), p.dim_x, p.dim_y, p.buf.pitch(),
                                        h_min, v_min, d_sd, l_px_row, l_px_col);

            glados::cuda::synchronize_stream(s.stream);
        }
        
    }
}
