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
 * Date: 04 December 2016
 * Authors: Jan Stephan <j.stephan@hzdr.de>
 */

#include <cmath>
#include <cstdint>

#include "backend.h"

namespace paris
{
    namespace openmp
    {
        namespace
        {
            auto generate_matrix(float* m, std::uint32_t dim_x, std::uint32_t dim_y, float h_min, float v_min,
                                 float d_sd, float l_px_row, float l_px_col) noexcept -> void
            {
                #pragma omp parallel for
                for(auto t = 0u; t < dim_y; ++t)
                {
                    #pragma omp simd
                    for(auto s = 0u; s < dim_x; ++s)
                    {
                        const auto coord = s + t * dim_x;

                        // prevent conversion warnings
                        const auto s_f = static_cast<float>(s);
                        const auto t_f = static_cast<float>(t);

                        // detector coordinates in mm
                        const auto h_s = (l_px_row / 2) + s_f * l_px_row + h_min;
                        const auto v_t = (l_px_col / 2) + t_f * l_px_col + v_min;

                        // calculate weight
                        m[coord] = d_sd / std::sqrt(d_sd * d_sd + h_s * h_s + v_t * v_t);
                    }
                }
            }
        }

        auto weight(projection_device_type& p, float h_min, float v_min, float d_sd, float l_px_row, float l_px_col)
            noexcept
            -> void
        {
            static auto matrix = projection_device_buffer_type{nullptr};
            if(matrix == nullptr)
            {
                matrix = std::make_unique<float[]>(p.dim_x * p.dim_y);
                generate_matrix(matrix.get(), p.dim_x, p.dim_y, h_min, v_min, d_sd, l_px_row, l_px_col);
            }

            #pragma omp parallel for schedule(dynamic)
            for(auto t = 0u; t < p.dim_y; ++t)
            {
                #pragma omp simd
                for(auto s = 0u; s < p.dim_x; ++s)
                {
                    const auto coord = s + t * p.dim_x;
                    p.buf[coord] *= matrix[coord];
                }
            }
        }
    }
}
