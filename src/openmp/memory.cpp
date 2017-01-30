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
 * Date: 04 December 2016
 * Authors: Jan Stephan <j.stephan@hzdr.de>
 */

#include <algorithm>
#include <memory>

#include "backend.h"
#include "../projection.h"

namespace paris
{
    namespace openmp
    {
        auto make_projection_host(std::uint32_t dim_x, std::uint32_t dim_y) -> projection_host_type
        {
            auto ptr = std::make_unique<float[]>(dim_x * dim_y);
            return projection<projection_host_buffer_type, metadata>{std::move(ptr), dim_x, dim_y, 0, 0.f, metadata{}}; 
        }

        auto make_projection_device(std::uint32_t dim_x, std::uint32_t dim_y) -> projection_device_type
        {
            return make_projection_host(dim_x, dim_y);
        }

        auto make_volume_host(std::uint32_t dim_x, std::uint32_t dim_y, std::uint32_t dim_z) -> volume_host_type
        {
            auto ptr = std::make_unique<float[]>(dim_x * dim_y * dim_z);
            // fill with 0
            std::fill_n(ptr.get(), dim_x * dim_y * dim_z, 0.f);
            return volume<volume_host_buffer_type>{std::move(ptr), dim_x, dim_y, dim_z, 0}; 
        }

        auto make_volume_device(std::uint32_t dim_x, std::uint32_t dim_y, std::uint32_t dim_z) -> volume_device_type
        {
            return make_volume_host(dim_x, dim_y, dim_z);
        }

        auto copy_h2d(const projection_host_type& h_p, projection_device_type& d_p) noexcept -> void
        {
            std::copy_n(h_p.buf.get(), h_p.dim_x * h_p.dim_y, d_p.buf.get());
            d_p.idx = h_p.idx;
            d_p.phi = h_p.phi;
            d_p.meta = h_p.meta;
        }

        auto copy_d2h(const projection_device_type& d_p, projection_host_type& h_p) noexcept -> void
        {
            copy_h2d(d_p, h_p);
        }

        auto copy_h2d(const volume_host_type& h_v, volume_device_type& d_v) noexcept -> void
        {
            std::copy_n(h_v.buf.get(), h_v.dim_x * h_v.dim_y * h_v.dim_z, d_v.buf.get());
            d_v.off = h_v.off;
        }

        auto copy_d2h(const volume_device_type& d_v, volume_host_type& h_v) noexcept -> void
        {
            copy_h2d(d_v, h_v);
        }
    }
}
