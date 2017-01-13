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
        namespace detail
        {
            auto do_weighting(float* out, const float* in, std::uint32_t dim_x, std::uint32_t dim_y,
                              float h_min, float v_min, float d_sd, float l_px_row, float l_px_col) noexcept -> void
            {
                #pragma omp parallel for collapse(2)
                for(auto t = 0u; t < dim_y; ++t)
                {
                    for(auto s = 0u; s < dim_x; ++s)
                    {
                        auto coord = s + t * dim_x;
                        auto val = in[coord];

                        // prevent conversion warnings
                        auto s_f = static_cast<float>(s);
                        auto t_f = static_cast<float>(t);

                        // detector coordinates in mm
                        auto h_s = (l_px_row / 2) + s_f * l_px_row + h_min;
                        auto v_t = (l_px_col / 2) + t_f * l_px_col + v_min;

                        // calculate weight
                        auto w_st = d_sd / std::sqrt(std::pow(d_sd, 2.f) + std::pow(h_s, 2.f) + std::pow(v_t, 2.f));

                        out[coord] = val * w_st;
                    }
                }
            }
        }
    }
}
