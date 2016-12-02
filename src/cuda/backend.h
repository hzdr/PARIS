/*
 * This file is part of the ddafa reconstruction program.
 *
 * Copyright (C) 2016 Helmholtz-Zentrum Dresden-Rossendorf
 *
 * ddafa is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ddafa is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with ddafa. If not, see <http://www.gnu.org/licenses/>.
 *
 * Date: 30 November 2016
 * Authors: Jan Stephan
 */

#ifndef DDAFA_CUDA_BACKEND_H_
#define DDAFA_CUDA_BACKEND_H_

#include <cstddef>

#include <ddrf/cuda/memory.h>

namespace ddafa
{
    namespace cuda
    {
        /*
         * TYPEDEFS
         * */
        using allocator = ddrf::cuda::device_allocator<float, ddrf::memory_layout::pointer_2D>;
        
        using host_ptr_1D = ddrf::cuda::pinned_host_ptr<float>;
        using host_ptr_2D = ddrf::cuda::pinned_host_ptr<float>;
        using host_ptr_3D = ddrf::cuda::pinned_host_ptr<float>;
        using device_ptr_1D = ddrf::cuda::device_ptr<float>;
        using device_ptr_2D = ddrf::cuda::pitched_device_ptr<float>;
        using device_ptr_3D = ddrf::cuda::pitched_device_ptr<float>;

        /*
         * Memory functions
         * */
        auto make_host_ptr(std::size_t n) -> host_ptr_1D;
        auto make_host_ptr(std::size_t x, std::size_t y) -> host_ptr_2D;
        auto make_host_ptr(std::size_t x, std::size_t y, std::size_t z) -> host_ptr_3D;

        auto make_device_ptr(std::size_t n) -> device_ptr_1D;
        auto make_device_ptr(std::size_t x, std::size_t y) -> device_ptr_2D;
        auto make_device_ptr(std::size_t x, std::size_t y, std::size_t z) -> device_ptr_3D;

        /*
         * Algorithms
         * */

    }
}

#endif /* DDAFA_CUDA_BACKEND_H_ */
