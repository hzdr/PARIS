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
 * Date: 30 November 2016
 * Authors: Jan Stephan <j.stephan@hzdr.de>
 */

#ifndef PARIS_CUDA_BACKEND_H_
#define PARIS_CUDA_BACKEND_H_

#include <cstdint>
#include <vector>

#include <cufft.h>

#include <glados/memory.h>
#include <glados/cuda/memory.h>

#include "../geometry.h"
#include "../projection.h"
#include "../region_of_interest.h"
#include "../subvolume_information.h"
#include "../volume.h"

namespace paris
{
    namespace cuda
    {
        namespace detail
        {
            using allocator = glados::cuda::device_allocator<float, glados::memory_layout::pointer_2D>;
            using pool = glados::pool_allocator<float, glados::memory_layout::pointer_2D, allocator>;
        }

        using projection_host_buffer_type = glados::cuda::pinned_host_ptr<float>;
        using projection_device_buffer_type = typename detail::pool::smart_pointer;
        using volume_host_buffer_type = glados::cuda::pinned_host_ptr<float>;
        using volume_device_buffer_type = glados::cuda::pitched_device_ptr<float>;

        struct cuda_stream
        {
            cuda_stream();
            ~cuda_stream();
            cudaStream_t stream;
        };

        using projection_host_type = projection<projection_host_buffer_type, cuda_stream>;
        using projection_device_type = projection<projection_device_buffer_type, cuda_stream>;
        using volume_host_type = volume<volume_host_buffer_type, cuda_stream>;
        using volume_device_type = volume<volume_device_buffer_type, cuda_stream>;

        auto make_projection_host(std::uint32_t dim_x, std::uint32_t dim_y) -> projection_host_type;
        auto make_projection_device(std::uint32_t dim_x, std::uint32_t dim_y) -> projection_device_type;

        auto make_volume_host(std::uint32_t dim_x, std::uint32_t dim_y, std::uint32_t dim_z) -> volume_host_type;
        auto make_volume_device(std::uint32_t dim_x, std::uint32_t dim_y, std::uint32_t dim_z) -> volume_device_type;

        auto copy_h2d(const projection_host_type& h_p, projection_device_type& d_p) -> void;
        auto copy_d2h(const projection_device_type& d_p, projection_host_type& h_p) -> void;

        auto copy_h2d(const volume_host_type& h_v, volume_device_type& d_v) -> void;
        auto copy_d2h(const volume_device_type& d_v, volume_host_type& h_v) -> void;

        auto make_subvolume_information(const volume_geometry& vol_geo, const detector_geometry& det_geo)
            -> subvolume_info;

        auto weight(projection_device_type& p, float h_min, float v_min, float d_sd, float l_px_row, float l_px_col)
            -> void;

        using filter_buffer_type = glados::cuda::device_ptr<cufftComplex>;
        auto make_filter(std::uint32_t size, float tau) -> filter_buffer_type;
        auto apply_filter(projection_device_type& p, const filter_buffer_type& k, std::uint32_t filter_size,
                          std::uint32_t n_col) -> void;

        auto backproject(projection_device_type& p, volume_device_type& v, std::uint32_t v_offset,
                         const detector_geometry& det_geo, const volume_geometry& vol_geo, 
                         bool enable_roi, const region_of_interest& roi,
                         float delta_s, float delta_t) -> void;

        /**
         * Device management
         * */
        using device_handle = int;
        auto get_devices() -> std::vector<device_handle>;
        auto set_device(device_handle& device) -> void;
    }
}

#endif /* PARIS_CUDA_BACKEND_H_ */
