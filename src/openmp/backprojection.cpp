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

#include "../region_of_interest.h"

#include "backend.h"

namespace paris
{
    namespace openmp
    {
        namespace
        {
            inline auto vol_centered_coordinate(std::uint32_t coord, std::uint32_t dim, float size) noexcept -> float
            {
                auto size2 = size / 2.f;
                return -(static_cast<float>(dim) * size2) + size2 + static_cast<float>(coord) * size;
            }

            inline auto proj_real_coordinate(float coord, std::uint32_t dim, float size, float offset) noexcept -> float
            {
                auto size2 = size / 2.f;
                auto min = -(static_cast<float>(dim) * size2) - offset;
                return (coord - min) / size - (1.f / 2.f);
            }

            auto interpolate(const float* p, float x, float y, std::uint32_t dim_x, std::uint32_t dim_y) noexcept
                -> float
            {
                auto x1 = std::floor(x);
                auto x2 = x1 + 1.f;
                auto y1 = std::floor(y);
                auto y2 = y1 + 1.f;

                auto x1u = static_cast<std::uint32_t>(x1);
                auto x2u = static_cast<std::uint32_t>(x2);
                auto y1u = static_cast<std::uint32_t>(y1);
                auto y2u = static_cast<std::uint32_t>(y2);

                auto x1_valid = x1 >= 0.f;
                auto x2_valid = x2 < static_cast<float>(dim_x);
                auto y1_valid = y1 >= 0.f;
                auto y2_valid = y2 < static_cast<float>(dim_y);

                auto interp = 0.f;
                if(x1_valid && x2_valid && y1_valid && y2_valid)
                {
                    auto q11 = p[x1u + y1u * dim_x];
                    auto q12 = p[x1u + y2u * dim_x];
                    auto q21 = p[x2u + y1u * dim_x];
                    auto q22 = p[x2u + y2u * dim_x];
                    auto interp_y1 = (x2 - x) / (x2 - x1) * q11 + (x - x1) / (x2 - x1) * q21;
                    auto interp_y2 = (x2 - x) / (x2 - x1) * q12 + (x - x1) / (x2 - x1) * q22;

                    interp = (y2 - y) / (y2 - y1) * interp_y1 + (y - y1) / (y2 - y1) * interp_y2;
                }

                return interp;
            }

            template <bool enable_roi>
            auto do_backprojection(float* vol_ptr, std::uint32_t v_dim_x, std::uint32_t v_dim_y, std::uint32_t v_dim_z,
                                   const float* p_ptr, std::uint32_t p_dim_x, std::uint32_t p_dim_y,
                                   std::uint32_t offset,
                                   std::uint32_t v_dim_x_full, std::uint32_t v_dim_y_full, std::uint32_t v_dim_z_full,
                                   float l_vx_x, float l_vx_y, float l_vx_z,
                                   float l_px_x, float l_px_y, float d_so, float d_sd, float delta_s, float delta_t,
                                   float sin, float cos, const region_of_interest& roi) noexcept -> void
            {
                #pragma omp parallel for collapse(3)
                for(auto m = 0u; m < v_dim_z; ++m)
                {
                    for(auto l = 0u; l < v_dim_y; ++l)
                    {
                        for(auto k = 0u; k < v_dim_x; ++k)
                        {
                            const auto coord = k + l * v_dim_x + m * v_dim_x * v_dim_y;

                            // add ROI offset -- this should get optimized away for enable_roi == false
                            if(enable_roi)
                            {
                                k += roi.x1;
                                l += roi.y1;
                                m += roi.z1;
                            }

                            // add offset for the current subvolume
                            m += offset;

                            // get centered coordinates -- volume center is at (0, 0, 0)
                            const auto x_k = vol_centered_coordinate(k, v_dim_x_full, l_vx_x);
                            const auto y_l = vol_centered_coordinate(l, v_dim_y_full, l_vx_y);
                            const auto z_m = vol_centered_coordinate(m, v_dim_z_full, l_vx_z);

                            // rotate coordinates
                            const auto s = x_k * cos + y_l * sin;
                            const auto t = -x_k * sin + y_l * cos;

                            // project rotated coordinates
                            const auto factor = d_sd / (s + d_so);
                            const auto h = proj_real_coordinate(t * factor,
                                                                p_dim_x,
                                                                l_px_x,
                                                                delta_s);
                            const auto v = proj_real_coordinate(z_m * factor,
                                                                p_dim_y,
                                                                l_px_y,
                                                                delta_t);
                            
                            // get projection value through interpolation
                            const auto det = interpolate(p_ptr, h, v, p_dim_x, p_dim_y);
                            
                            // backproject
                            const auto u = -(d_so / (s + d_so));
                            vol_ptr[coord] += 0.5f * det * u * u;

                            // restore old coordinates
                            m -= offset;
                            if(enable_roi)
                            {
                                k -= roi.x1;
                                l -= roi.y1;
                                m -= roi.z1;
                            }
                        }
                    }
                }
            }
        }

        auto backproject(const projection_device_type& p, volume_device_type& v, std::uint32_t v_offset,
                         const detector_geometry& det_geo, const volume_geometry& vol_geo,
                         bool enable_roi, const region_of_interest& roi, float sin, float cos,
                         float delta_s, float delta_t) noexcept -> void
        {
            // constants for the backprojection - these never change
            static const auto v_dim_x_full = vol_geo.dim_x;
            static const auto v_dim_y_full = vol_geo.dim_y;
            static const auto v_dim_z_full = vol_geo.dim_z;

            static const auto l_vx_x = vol_geo.l_vx_x;
            static const auto l_vx_y = vol_geo.l_vx_y;
            static const auto l_vx_z = vol_geo.l_vx_z;

            static const auto l_px_x = det_geo.l_px_row;
            static const auto l_px_y = det_geo.l_px_col;

            static const auto d_s = delta_s;
            static const auto d_t = delta_t;

            static const auto d_so = det_geo.d_so;
            static const auto d_sd = std::abs(det_geo.d_so) + std::abs(det_geo.d_od);

            // variable for the backprojection - this might change between subvolumes
            thread_local static auto offset = v_offset;

            // backproject and apply ROI as needed
            if(enable_roi)
                do_backprojection<true>(v.buf.get(), v.dim_x, v.dim_y, v.dim_z,
                                        p.buf.get(), p.dim_x, p.dim_y,
                                        offset,
                                        v_dim_x_full, v_dim_y_full, v_dim_z_full,
                                        l_vx_x, l_vx_y, l_vx_z,
                                        l_px_x, l_px_y, d_so, d_sd, d_s, d_t,
                                        sin, cos, roi);
            else
                do_backprojection<false>(v.buf.get(), v.dim_x, v.dim_y, v.dim_z,
                                         p.buf.get(), p.dim_x, p.dim_y,
                                         offset,
                                         v_dim_x_full, v_dim_y_full, v_dim_z_full,
                                         l_vx_x, l_vx_y, l_vx_z,
                                         l_px_x, l_px_y, d_so, d_sd, d_s, d_t,
                                         sin, cos, roi);
        }
    }
}
