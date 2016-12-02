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
 * Authors: Jan Stephan <j.stephan@hzdr.de>
 */

#ifndef DDAFA_CUDA_BACKEND_H_
#define DDAFA_CUDA_BACKEND_H_

#include <cstddef>
#include <utility>

#ifndef __NVCC__
#include <cuda_runtime.h>
#endif

#include <ddrf/cuda/algorithm.h>
#include <ddrf/cuda/memory.h>
#include <ddrf/cuda/sync_policy.h>

namespace ddafa
{
    namespace cuda
    {
        /*
         * Runtime management
         * */
        using error_type = cudaError_t;
        constexpr auto success = cudaSuccess;
        auto print_error(const std::string& msg, error_type err) noexcept -> void;

        // exceptions
        using bad_alloc = ddrf::cuda::bad_alloc;
        using invalid_argument = ddrf::cuda::invalid_argument;
        using runtime_error = ddrf::cuda::runtime_error;

        /*
         * Memory management
         * */
        using allocator = ddrf::cuda::device_allocator<float, ddrf::memory_layout::pointer_2D>;
        
        using host_ptr_1D = ddrf::cuda::pinned_host_ptr<float>;
        using host_ptr_2D = ddrf::cuda::pinned_host_ptr<float>;
        using host_ptr_3D = ddrf::cuda::pinned_host_ptr<float>;
        using device_ptr_1D = ddrf::cuda::device_ptr<float>;
        using device_ptr_2D = ddrf::cuda::pitched_device_ptr<float>;
        using device_ptr_3D = ddrf::cuda::pitched_device_ptr<float>;

        auto make_host_ptr(std::size_t n) -> host_ptr_1D;
        auto make_host_ptr(std::size_t x, std::size_t y) -> host_ptr_2D;
        auto make_host_ptr(std::size_t x, std::size_t y, std::size_t z) -> host_ptr_3D;

        auto make_device_ptr(std::size_t n) -> device_ptr_1D;
        auto make_device_ptr(std::size_t x, std::size_t y) -> device_ptr_2D;
        auto make_device_ptr(std::size_t x, std::size_t y, std::size_t z) -> device_ptr_3D;

        /*
         * Device management
         * */
        using device_handle = int;

        auto set_device(const device_handle& device) -> void;
        auto set_device_noexcept(const device_handle& device) noexcept -> error_type; 

        /*
         * Synchronization
         * */
        constexpr auto sync = ddrf::cuda::sync;
        constexpr auto async = ddrf::cuda::async;

        using async_handle = cudaStream_t;
        auto make_async_handle() -> async_handle;

        auto synchronize(async_handle& handle) -> void;

        /*
         * Basic algorithms
         * */
        template <class SyncPolicy, class D, class S, class... Args>
        auto copy(SyncPolicy&& policy, D& dst, const S& src, Args&&... args) -> void
        {
            ddrf::cuda::copy(std::forward<SyncPolicy>(policy), dst, src, std::forward<Args>(args)...);
        }

        template <class SyncPolicy, class P, class... Args>
        auto fill(SyncPolicy&& policy, P& p, int value, Args&&... args) -> void
        {
            ddrf::cuda::fill(std::forward<SyncPolicy>(policy), p, value, std::forward<Args>(args)...);
        }
    }
}

#endif /* DDAFA_CUDA_BACKEND_H_ */
