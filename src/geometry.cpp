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
 * Date: 27 Oktober 2016
 * Authors: Jan Stephan
 */

#include <cstdint>
#include <cmath>
#include <iomanip>

#include <boost/log/trivial.hpp>

#include "geometry.h"
#include "region_of_interest.h"

namespace ddafa
{
    namespace
    {
        auto make_volume_geometry(const detector_geometry& det_geo) noexcept -> volume_geometry
        {
            auto vol_geo = volume_geometry{};

            /* Casts and removing some boilerplate code from the calculations below*/
            auto n_row = static_cast<float>(det_geo.n_row);
            auto l_px_row = det_geo.l_px_row;
            auto delta_s = std::abs(det_geo.delta_s * l_px_row); // the offset in det_geo is measured in pixels!

            auto n_col = static_cast<float>(det_geo.n_col);
            auto l_px_col = det_geo.l_px_col;
            auto delta_t = std::abs(det_geo.delta_t * l_px_col);

            auto d_so = std::abs(det_geo.d_so);
            auto d_sd = std::abs(det_geo.d_od) + d_so;

            /* Calculate slice dimensions */
            auto alpha = std::atan((((n_row * l_px_row) / 2.f) + delta_s) / d_sd);
            auto r = d_so * std::sin(alpha);

            vol_geo.l_vx_x = r / ((((n_row * l_px_row) / 2.f) + delta_s) / l_px_row);
            vol_geo.l_vx_y = vol_geo.l_vx_x;

            vol_geo.dim_x = static_cast<std::uint32_t>((2.f * r) / vol_geo.l_vx_x);
            vol_geo.dim_y = vol_geo.dim_x;

            /* Calculate number of slices */
            vol_geo.l_vx_z = vol_geo.l_vx_x;
            vol_geo.dim_z = static_cast<std::uint32_t>(((n_col * l_px_col / 2.f) + delta_t) * (d_so / d_sd) * (2.f / vol_geo.l_vx_z));

            return vol_geo;
        }

        auto apply_roi(volume_geometry& vol_geo,
                        std::uint32_t x1, std::uint32_t x2,
                        std::uint32_t y1, std::uint32_t y2,
                        std::uint32_t z1, std::uint32_t z2) noexcept -> bool
        {
            auto check_coords = [](std::uint32_t low, std::uint32_t high) { return low < high; };
            auto check_dims = [](std::uint32_t updated, std::uint32_t old) { return updated <= old; };

            if(check_coords(x1, x2) && check_coords(y1, y2) && check_coords(z1, z2))
            {
                auto dim_x = x2 - x1;
                auto dim_y = y2 - y1;
                auto dim_z = z2 - z1;
                if(check_dims(dim_x, vol_geo.dim_x) && check_dims(dim_y, vol_geo.dim_y) && check_dims(dim_z, vol_geo.dim_z))
                {
                    vol_geo.dim_x = dim_x;
                    vol_geo.dim_y = dim_y;
                    vol_geo.dim_z = dim_z;
                }
                else
                {
                    BOOST_LOG_TRIVIAL(warning) << "New volume dimensions exceed old volume dimensions. ROI NOT applied.";
                    return false;
                }
            }
            else
            {
                BOOST_LOG_TRIVIAL(warning) << "Invalid ROI coordinates. ROI NOT applied.";
                return false;
            }

            return true;
        }
    }

    auto calculate_volume_geometry(const detector_geometry& det_geo, bool enable_roi,
                                    std::uint32_t roi_x1, std::uint32_t roi_x2,
                                    std::uint32_t roi_y1, std::uint32_t roi_y2,
                                    std::uint32_t roi_z1, std::uint32_t roi_z2) noexcept -> volume_geometry
    {
        auto vol_geo = make_volume_geometry(det_geo);

        auto dim_x_mm = static_cast<float>(vol_geo.dim_x) * vol_geo.l_vx_x;
        auto dim_y_mm = static_cast<float>(vol_geo.dim_y) * vol_geo.l_vx_y;
        auto dim_z_mm = static_cast<float>(vol_geo.dim_z) * vol_geo.l_vx_z;

        BOOST_LOG_TRIVIAL(info) << "Volume dimensions [vx]: " << vol_geo.dim_x << " x " << vol_geo.dim_y << " x " << vol_geo.dim_z;
        BOOST_LOG_TRIVIAL(info) << "Volume dimensions [mm]: " << dim_x_mm << " x " << dim_y_mm  << " x " << dim_z_mm;

        if(enable_roi)
        {
            if(apply_roi(vol_geo, roi_x1, roi_x2, roi_y1, roi_y2, roi_z1, roi_z2))
            {
                dim_x_mm = static_cast<float>(vol_geo.dim_x) * vol_geo.l_vx_x;
                dim_y_mm = static_cast<float>(vol_geo.dim_y) * vol_geo.l_vx_y;
                dim_z_mm = static_cast<float>(vol_geo.dim_z) * vol_geo.l_vx_z;

                BOOST_LOG_TRIVIAL(info) << "Applied region of interest.";
                BOOST_LOG_TRIVIAL(info) << "Updated volume dimensions [vx]: " << vol_geo.dim_x << " x " << vol_geo.dim_y << " x " << vol_geo.dim_z;
                BOOST_LOG_TRIVIAL(info) << "Updated volume dimensions [mm]: " << dim_x_mm << " x " << dim_y_mm  << " x " << dim_z_mm;
            }
        }

        BOOST_LOG_TRIVIAL(info) << "Voxel size [mm]: " << std::setprecision(4) << vol_geo.l_vx_x << " x " << vol_geo.l_vx_y << " x " << vol_geo.l_vx_z;

        return vol_geo;
    }
}


