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

    auto calculate_volume_geometry(const detector_geometry& det_geo, bool enable_roi,
                                    std::uint32_t roi_x1, std::uint32_t roi_x2,
                                    std::uint32_t roi_y1, std::uint32_t roi_y2,
                                    std::uint32_t roi_z1, std::uint32_t roi_z2) noexcept -> volume_geometry;
}

#endif /* DDAFA_GEOMETRY_H_ */
