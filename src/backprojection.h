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
 * Date: 18 August 2016
 * Authors: Jan Stephan <j.stephan@hzdr.de>
 */

#ifndef PARIS_BACKPROJECTION_H_
#define PARIS_BACKPROJECTION_H_

#include <cstdint>

#include "backend.h"
#include "geometry.h"
#include "projection.h"
#include "region_of_interest.h"
#include "volume.h"

namespace paris
{
    auto backproject(backend::projection_device_type& p,
                     backend::volume_device_type& v,
                     std::uint32_t v_offset,
                     const detector_geometry& det_geo,
                     const volume_geometry& vol_geo,
                     bool enable_angles,
                     bool enable_roi,
                     const region_of_interest& roi)
        noexcept(true && noexcept(backend::backproject))
        -> void;
}

#endif /* PARIS_BACKPROJECTION_H_ */
