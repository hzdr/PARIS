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

#include "../geometry.h"
#include "../subvolume_information.h"
#include "backend.h"

namespace paris
{
    namespace openmp
    {
        auto make_subvolume_information(const volume_geometry& vol_geo, const detector_geometry& /* det_geo */) noexcept
            -> subvolume_info
        {
            auto subvol_info = subvolume_info{};

            subvol_info.geo.dim_x = vol_geo.dim_x;
            subvol_info.geo.dim_y = vol_geo.dim_y;
            subvol_info.geo.dim_z = vol_geo.dim_z;
            subvol_info.geo.remainder = 0u;
            subvol_info.num = 1;

            return subvol_info;
        }
    }
}
