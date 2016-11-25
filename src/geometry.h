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
 * Date: 18 August 2016
 * Authors: Jan Stephan
 */

#ifndef DDAFA_GEOMETRY_H_
#define DDAFA_GEOMETRY_H_

#include <cstdint>

namespace ddafa
{
    struct detector_geometry
    {
        // Detector
        std::uint32_t n_row;    // pixels per row
        std::uint32_t n_col;    // pixels per column
        float l_px_row;         // pixel size in horizontal direction [mm]
        float l_px_col;         // pixel size in vertical direction [mm]
        float delta_s;          // offset in horizontal direction [px]
        float delta_t;          // offset in vertical direction [px]

        // Support
        float d_so;             // distance source->object
        float d_od;             // distance object->detector

        // Rotation
        float delta_phi;        // difference between two successive angles - ignored if there is an angle file
    };

    struct volume_geometry
    {
        std::uint32_t dim_x;    // voxels in x direction
        std::uint32_t dim_y;    // voxels in y direction
        std::uint32_t dim_z;    // voxels in z direction

        float l_vx_x;           // voxel size in x direction [mm]
        float l_vx_y;           // voxel size in y direction [mm]
        float l_vx_z;           // voxel size in z direction [mm]
    };

    struct subvolume_geometry
    {
        std::uint32_t dim_x;
        std::uint32_t dim_y;
        std::uint32_t dim_z;

        /* Not all volumes are evenly distributable. The lowest subvolume therefore contains the remaining
         * slices.
         */
        std::uint32_t remainder;
    };

    auto calculate_volume_geometry(const detector_geometry& det_geo) noexcept -> volume_geometry;

    auto apply_roi(const volume_geometry& vol_geo, std::uint32_t roi_x1, std::uint32_t roi_x2,
                                                    std::uint32_t roi_y1, std::uint32_t roi_y2,
                                                    std::uint32_t roi_z1, std::uint32_t roi_z2) noexcept -> volume_geometry;
}

#endif /* DDAFA_GEOMETRY_H_ */
