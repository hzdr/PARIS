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

#include <cmath>
#include <cstdint>

#include <boost/log/trivial.hpp>

#include "backend.h"
#include "backprojection.h"
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
        -> void
    {
        // the following constants are global -> initialise once
        static const auto delta_s = det_geo.delta_s * det_geo.l_px_row;
        static const auto delta_t = det_geo.delta_t * det_geo.l_px_col;

        // get angular position of the current projection
        if(!enable_angles)
            p.phi = static_cast<float>(p.idx) * det_geo.delta_phi;

        // transform to radians
        p.phi *= static_cast<float>(M_PI) / 180.f;

        if(p.idx % 10u == 0u && p.valid)
            BOOST_LOG_TRIVIAL(info) << "Processing projection #" << p.idx;

        backend::backproject(p, v, v_offset, det_geo, vol_geo, enable_roi, roi, delta_s, delta_t);
    }
}
