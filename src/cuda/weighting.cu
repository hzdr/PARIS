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
            __global__ void matrix_generation_kernel(float* m, std::uint32_t dim_x, std::uint32_t dim_y,
                                                      std::size_t pitch, float h_min, float v_min, float d_sd,
                                                      float l_px_row, float l_px_col)
            {
                auto s = glados::cuda::coord_x();
                auto t = glados::cuda::coord_y();

                if((s < dim_x) && (t < dim_y))
                {
                    auto row = reinterpret_cast<float*>(reinterpret_cast<char*>(m) + t * pitch);

                    // detector coordinates in mm
                    const auto h_s = (l_px_row / 2.f) + s * l_px_row + h_min;
                    const auto v_t = (l_px_col / 2.f) + t * l_px_col + v_min;

                    // calculate weight
                    row[s] = d_sd * rsqrtf(d_sd * d_sd + h_s * h_s + v_t * v_t);
                }
            }

            __global__ void weighting_kernel(float* p, const float* m, std::uint32_t dim_x, std::uint32_t dim_y,
                                             std::size_t p_pitch, std::size_t m_pitch)
            {
                auto s = glados::cuda::coord_x();
                auto t = glados::cuda::coord_y();

                if((s < dim_x) && (t < dim_y))
                {
                    auto p_row = reinterpret_cast<float*>(reinterpret_cast<char*>(p) + t * p_pitch);
                    auto m_row = reinterpret_cast<const float*>(reinterpret_cast<const char*>(m) + t * m_pitch);

                    // write value
                    p_row[s] *= m_row[s];
                }
            }

        } 

        auto weight(projection_device_type& p, float h_min, float v_min, float d_sd, float l_px_row, float l_px_col)
            -> void
        {
            thread_local static auto matrix = glados::cuda::pitched_device_ptr<float>(nullptr);
            thread_local static auto generated_matrix = false;
            if(!generated_matrix)
            {
                matrix = glados::cuda::make_unique_device<float>(p.dim_x, p.dim_y);
                glados::cuda::launch_async(p.meta.stream, p.dim_x, p.dim_y,
                                           matrix_generation_kernel,
                                           matrix.get(), p.dim_x, p.dim_y, matrix.pitch(), h_min, v_min, d_sd,
                                           l_px_row, l_px_col);
                generated_matrix = true;
            }

            glados::cuda::launch_async(p.meta.stream, p.dim_x, p.dim_y,
                                        weighting_kernel,
                                        p.buf.get(), static_cast<const float*>(matrix.get()),
                                        p.dim_x, p.dim_y, p.buf.pitch(), matrix.pitch());
        }
        
    }
}
