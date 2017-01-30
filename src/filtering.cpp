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

#include "backend.h"
#include "filtering.h"
#include "geometry.h"

namespace paris
{
    auto filter(backend::projection_device_type& p, const detector_geometry& det_geo)
        noexcept(true && noexcept(backend::apply_filter))
        -> void
    {
        // the following variables are static and global -> initialise once
        static const auto filter_size = static_cast<std::uint32_t>(2 * std::pow(2.f, std::ceil(std::log2(det_geo.n_row))));
        static const auto n_col = det_geo.n_col;
        static const auto tau = det_geo.l_px_row;

        // the following variable is static and thread local -> initialise once per thread (= device)
        thread_local static const auto k = backend::make_filter(filter_size, tau);

        backend::apply_filter(p, k, filter_size, n_col);
    }
}
