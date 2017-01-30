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
 * Date: 27 January 2017
 * Authors: Jan Stephan <j.stephan@hzdr.de>
 */

#include "backend.h"
#include "geometry.h"
#include "make_volume.h"
#include "volume.h"

namespace paris
{
    auto make_volume(const subvolume_geometry& subvol_geo, bool last) -> backend::volume_device_type
    {
        auto dim_z = subvol_geo.dim_z;
        if(last)
            dim_z += subvol_geo.remainder;

        return backend::make_volume_device(subvol_geo.dim_x, subvol_geo.dim_y, dim_z);
    }
}

