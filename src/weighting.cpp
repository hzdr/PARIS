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

#include "backend.h"
#include "geometry.h"
#include "projection.h"
#include "weighting.h"

namespace paris
{
    auto weight(backend::projection_device_type& p, const detector_geometry& det_geo)
        noexcept(true && noexcept(backend::weight))
        -> void
    {
        if(!p.valid)
            return;

        // the following variables are static and valid for the entire program -> initialise once
        static const auto n_row_f = static_cast<float>(det_geo.n_row);
        static const auto n_col_f = static_cast<float>(det_geo.n_col);

        static const auto h_min = (det_geo.delta_s * det_geo.l_px_row) - ((n_row_f * det_geo.l_px_row) / 2);
        static const auto v_min = (det_geo.delta_t * det_geo.l_px_col) - ((n_col_f * det_geo.l_px_col) / 2);
        static const auto d_sd = std::abs(det_geo.d_so) + std::abs(det_geo.d_od);

        backend::weight(p, h_min, v_min, d_sd, det_geo.l_px_row, det_geo.l_px_col);
    }
}
