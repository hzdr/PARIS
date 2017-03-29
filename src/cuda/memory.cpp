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
 * Date: 28 January 2017
 * Authors: Jan Stephan <j.stephan@hzdr.de>
 */

#include <thread>

#include <glados/cuda/algorithm.h>
#include <glados/cuda/memory.h>
#include <glados/cuda/sync_policy.h>
#include <glados/cuda/utility.h>

#include "backend.h"

namespace paris
{
    namespace cuda
    {
        auto make_projection_host(std::uint32_t dim_x, std::uint32_t dim_y) -> projection_host_type
        {
            auto ptr = glados::cuda::make_unique_pinned_host<float>(dim_x, dim_y);
            return projection_host_type{std::move(ptr), dim_x, dim_y, 0u, 0.f, true, cuda_stream{}};
        }

        auto make_projection_device(std::uint32_t dim_x, std::uint32_t dim_y) -> projection_device_type
        {
            thread_local static auto allocator = detail::pool{5};
            auto ptr = allocator.allocate_smart(dim_x, dim_y);
            return projection_device_type{std::move(ptr), dim_x, dim_y, 0u, 0.f, true, cuda_stream{}};
        }

        auto make_volume_host(std::uint32_t dim_x, std::uint32_t dim_y, std::uint32_t dim_z) -> volume_host_type
        {
            auto ptr = glados::cuda::make_unique_pinned_host<float>(dim_x, dim_y, dim_z);
            return volume_host_type{std::move(ptr), dim_x, dim_y, dim_z, 0u, volume_metadata{}};
        }

        auto make_volume_device(std::uint32_t dim_x, std::uint32_t dim_y, std::uint32_t dim_z) -> volume_device_type
        {
            auto ptr = glados::cuda::make_unique_device<float>(dim_x, dim_y, dim_z);
            glados::cuda::fill(glados::cuda::sync, ptr, 0, dim_x, dim_y, dim_z);
            return volume_device_type{std::move(ptr), dim_x, dim_y, dim_z, 0u, volume_metadata{}};
        }

        auto copy_h2d(const projection_host_type& h_p, projection_device_type& d_p) -> void
        {
            glados::cuda::copy(glados::cuda::async, d_p.buf, h_p.buf, d_p.meta.stream, h_p.dim_x, h_p.dim_y);
            d_p.idx = h_p.idx;
            d_p.phi = h_p.phi;
        }

        auto copy_d2h(const projection_device_type& d_p, projection_host_type& h_p) -> void
        {
            glados::cuda::copy(glados::cuda::async, h_p.buf, d_p.buf, d_p.meta.stream, d_p.dim_x, d_p.dim_y);
            h_p.idx = d_p.idx;
            h_p.phi = d_p.phi;
        }

        auto copy_h2d(const volume_host_type& h_v, volume_device_type& d_v) -> void
        {
            if(h_v.meta.done_future.valid())
                h_v.meta.done_future.wait();

            glados::cuda::copy(glados::cuda::async, d_v.buf, h_v.buf, d_v.meta.s.stream, h_v.dim_x, h_v.dim_y, h_v.dim_z);
            d_v.off = h_v.off;
        }

        auto copy_d2h(const volume_device_type& d_v, volume_host_type& h_v) -> void
        {
            if(d_v.meta.done_future.valid())
                d_v.meta.done_future.wait();

            glados::cuda::copy(glados::cuda::async, h_v.buf, d_v.buf, d_v.meta.s.stream, d_v.dim_x, d_v.dim_y, d_v.dim_z);
            glados::cuda::synchronize_stream(d_v.meta.s.stream);
            h_v.off = d_v.off;
        }
    }
}
