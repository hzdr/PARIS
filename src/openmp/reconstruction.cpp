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
 * Date: 19 December 2016
 * Authors: Jan Stephan <j.stephan@hzdr.de>
 */

#include <cmath>
#include <cstdint>
#include <type_traits>

#include <boost/log/trivial.hpp>

#include "../reconstruction_constants.h"
#include "../region_of_interest.h"

#include "backend.h"

namespace paris
{
    namespace openmp
    {
        namespace
        {
            reconstruction_constants consts__;
            region_of_interest roi__;

            inline auto vol_centered_coordinate(std::uint32_t coord, std::uint32_t dim, float size) noexcept -> float
            {
                auto size2 = size / 2.f;
                return -(dim * size2) + size2 + coord * size;
            }

            inline auto proj_real_coordinate(float coord, std::uint32_t dim, float size, float offset) noexcept -> float
            {
                auto size2 = size / 2.f;
                auto min = -(dim * size2) - offset;
                return (coord - min) / size - (1.f / 2.f);
            }

            auto interpolate(const float* p, float x, float y, std::uint32_t dim_x, std::uint32_t dim_y) noexcept -> float
            {
                auto x1 = static_cast<std::int32_t>(std::floor(x));
                auto x2 = x1 + 1;
                auto y1 = static_cast<std::int32_t>(std::floor(y));
                auto y2 = y1 + 1;

                auto x1_valid = x1 >= 0;
                auto x2_valid = x2 < dim_x;
                auto y1_valid = y1 >= 0;
                auto y2_valid = y2 < dim_y;

                auto interp = 0.f;
                if(x1_valid && x2_valid && y1_valid && y2_valid)
                {
                    auto q11 = p[x1 + y1 * dim_x];
                    auto q12 = p[x1 + y2 * dim_x];
                    auto q21 = p[x2 + y1 * dim_x];
                    auto q22 = p[x2 + y2 * dim_x];
                    auto interp_y1 = (x2 - x) / (x2 - x1) * q11 + (x - x1) / (x2 - x1) * q21;
                    auto interp_y2 = (x2 - x) / (x2 - x1) * q12 + (x - x1) / (x2 - x1) * q22;

                    interp = (y2 - y) / (y2 - y1) * interp_y1 + (y - y1) / (y2 - y1) * interp_y2;
                }

                return interp;
            }
        }

        auto set_reconstruction_constants(const reconstruction_constants& rc) noexcept -> error_type
        {
            consts__ = rc;
            return success;
        }

        auto set_roi(const region_of_interest& roi) noexcept -> error_type
        {
            roi__ = roi;
            return success;
        }

        namespace detail
        {
            auto do_backprojection(float* vol_ptr, std::uint32_t dim_x, std::uint32_t dim_y, std::uint32_t dim_z,
                                   const float* p_ptr, std::uint32_t p_dim_x, std::uint32_t p_dim_y,
                                   float sin, float cos, bool enable_roi) noexcept -> void
            {
                #pragma omp parallel for collapse(3)
                for(auto m = 0u; m < dim_z; ++m)
                {
                    for(auto l = 0u; l < dim_y; ++l)
                    {
                        for(auto k = 0u; k < dim_x; ++k)
                        {
                            auto coord = k + l * dim_x + m * dim_x * dim_y;

                            // add ROI offset
                            if(enable_roi)
                            {
                                k += roi__.x1;
                                l += roi__.y1;
                                m += roi__.z1;
                            }

                            // add offset for the current subvolume
                            m += consts__.vol_offset;

                            // get centered coordinates -- volume center is at (0, 0, 0)
                            auto x_k = vol_centered_coordinate(k, consts__.vol_dim_x_full, consts__.l_vx_x);
                            auto y_l = vol_centered_coordinate(l, consts__.vol_dim_y_full, consts__.l_vx_y);
                            auto z_m = vol_centered_coordinate(m, consts__.vol_dim_z_full, consts__.l_vx_z);

                            // rotate coordinates
                            auto s = x_k * cos + y_l * sin;
                            auto t = -x_k * sin + y_l * cos;

                            // project rotated coordinates
                            auto factor = consts__.d_sd / (s + consts__.d_so);
                            auto h = proj_real_coordinate(t * factor,
                                                          consts__.proj_dim_x,
                                                          consts__.l_px_x,
                                                          consts__.delta_s);
                            auto v = proj_real_coordinate(z_m * factor,
                                                          consts__.proj_dim_y,
                                                          consts__.l_px_x,
                                                          consts__.delta_t);
                            
                            // get projection value through interpolation
                            auto det = interpolate(p_ptr, h, v, p_dim_x, p_dim_y);
                            
                            // backproject
                            auto u = -(consts__.d_so / (s + consts__.d_so));
                            vol_ptr[coord] += 0.5f * det * std::pow(u, 2.f);

                            // restore old coordinates
                            m -= consts__.vol_offset;
                            if(enable_roi)
                            {
                                k -= roi__.x1;
                                l -= roi__.y1;
                                m -= roi__.z1;
                            }
                        }
                    }
                }
            }
        }
    }
}
