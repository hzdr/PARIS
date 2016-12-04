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
#include <cstdint>
#include <utility>
#include <vector>

#ifndef __NVCC__
#include <cuda_runtime.h>
#endif

#include <ddrf/cuda/algorithm.h>
#include <ddrf/cuda/memory.h>
#include <ddrf/cuda/sync_policy.h>
#include <ddrf/cufft/plan.h>

#include "../reconstruction_constants.h"
#include "../region_of_interest.h"
#include "../subvolume_information.h"

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

        constexpr auto name = "CUDA";
        /*
         * Memory management
         * */
        using allocator = ddrf::cuda::device_allocator<float, ddrf::memory_layout::pointer_2D>;
        
        template <class T>
        using host_ptr_1D = ddrf::cuda::pinned_host_ptr<T>;

        template <class T>
        using host_ptr_2D = ddrf::cuda::pinned_host_ptr<T>;

        template <class T>
        using host_ptr_3D = ddrf::cuda::pinned_host_ptr<T>;

        template <class T>
        using device_ptr_1D = ddrf::cuda::device_ptr<T>;

        template <class T>
        using device_ptr_2D = ddrf::cuda::pitched_device_ptr<T>;

        template <class T>
        using device_ptr_3D = ddrf::cuda::pitched_device_ptr<T>;

        template <class T>
        auto make_host_ptr(std::size_t n) -> host_ptr_1D<T>
        {
            return ddrf::cuda::make_unique_pinned_host<T>(n);
        }

        template <class T>
        auto make_host_ptr(std::size_t x, std::size_t y) -> host_ptr_2D<T>
        {
            return ddrf::cuda::make_unique_pinned_host<T>(x, y);
        }

        template <class T>
        auto make_host_ptr(std::size_t x, std::size_t y, std::size_t z) -> host_ptr_3D<T>
        {
            return ddrf::cuda::make_unique_pinned_host<T>(x, y, z);
        }

        template <class T>
        auto make_device_ptr(std::size_t n) -> device_ptr_1D<T>
        {
            return ddrf::cuda::make_unique_device<T>(n);
        }

        template <class T>
        auto make_device_ptr(std::size_t x, std::size_t y) -> device_ptr_2D<T>
        {
            return ddrf::cuda::make_unique_device<T>(x, y);
        }

        template <class T>
        auto make_device_ptr(std::size_t x, std::size_t y, std::size_t z) -> device_ptr_3D<T>
        {
            return ddrf::cuda::make_unique_device<T>(x, y, z);
        }

        /*
         * Device management
         * */
        using device_handle = int;

        auto set_device(const device_handle& device) -> void;
        auto set_device_noexcept(const device_handle& device) noexcept -> error_type; 
        
        auto get_devices() -> std::vector<device_handle>;

        /*
         * Synchronization
         * */
        constexpr auto sync = ddrf::cuda::sync;
        constexpr auto async = ddrf::cuda::async;

        using async_handle = cudaStream_t;
        constexpr auto default_async_handle = cudaStream_t{0};

        auto make_async_handle() -> async_handle;
        auto destroy_async_handle(async_handle& handle) noexcept -> error_type;

        auto synchronize(const async_handle& handle) -> void;

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

        /*
         * Subvolume information
         * */
        /*
         * Based on the target volume dimensions, the detector geometry and the number of projections in the pipeline
         * this function creates subvolume geometries according to the following algorithm:
         *
         * 1) Divide the volume by the number of available devices
         * 2) Check if the subvolume and proj_num projections fit into device memory
         *      a) if yes, return
         *      b) else, divide the volume by 2 until it fits
         */
        auto make_subvolume_information(const volume_geometry& vol_geo, const detector_geometry& det_geo, int proj_num) -> subvolume_info;

        /*
         * Weighting
         * */
        namespace detail
        {
            auto call_weighting_kernel(float* dst, const float* src, std::uint32_t width, std::uint32_t height, std::size_t pitch,
                                       float h_min, float v_min, float d_sd, float l_px_row, float l_px_col, async_handle handle) -> void;
        }

        template <class Proj>
        auto weight(Proj& p, float h_min, float v_min, float d_sd, float l_px_row, float l_px_col) -> void
        {
            detail::call_weighting_kernel(p.ptr.get(), p.ptr.get(),
                                          p.width, p.height, p.ptr.pitch(),
                                          h_min, v_min, d_sd, l_px_row, l_px_col,
                                          p.async_handle);
        }

        /*
         * Filtering
         * */
        namespace fft
        {
            constexpr auto name = "cuFFT";

            using bad_alloc = ddrf::cufft::bad_alloc;
            using invalid_argument = ddrf::cufft::invalid_argument;
            using runtime_error = ddrf::cufft::runtime_error;

            using complex_type = cufftComplex;
            using transformation_type = cufftType;
            constexpr auto r2c = CUFFT_R2C;
            constexpr auto c2r = CUFFT_C2R;

            template <transformation_type Type>
            using plan_type = ddrf::cufft::plan<Type>;

            // cuFFT doesn't need pointers to the data for plan creation -> ignore in and out ptrs
            template <transformation_type Type, class Src, class Dst>
            auto make_plan(int rank, int *n, int batch_size,
                           Src* /* in */, int* inembed, int istride, int idist,
                           Dst* /* out */, int* onembed, int ostride, int odist)
            -> plan_type<Type>
            {
                return ddrf::cufft::plan<Type>{rank, n,
                                               inembed, istride, idist,
                                               onembed, ostride, odist,
                                               batch_size};
            }
        }

        auto make_filter(std::uint32_t size, float tau) -> device_ptr_1D<fft::complex_type>;

        template <class Ptr>
        auto calculate_distance(Ptr& p) -> int
        {
            return static_cast<int>(p.pitch() / sizeof(typename Ptr::element_type));
        }

        template <class In, class Out>
        auto expand(const In& in, Out& out, std::uint32_t x, std::uint32_t y) -> void
        {
            // reset expanded projection
            ddrf::cuda::fill(ddrf::cuda::async, out, 0, in.async_handle, x, y);

            // copy original projection to expanded projection
            ddrf::cuda::copy(ddrf::cuda::async, out, in.ptr, in.async_handle, in.width, in.height);
        }

        template <class In, class Out, class Plan>
        auto transform(const In& in, Out& out, Plan& plan, async_handle& handle) -> void
        {
            plan.set_stream(handle);
            plan.execute(in.get(), out.get());
        }

        namespace detail
        {
            auto call_filter_application_kernel(async_handle& handle,
                                                fft::complex_type* in,
                                                const fft::complex_type* filter,
                                                std::uint32_t dim_x, std::uint32_t dim_y,
                                                std::size_t in_pitch) -> void;
        }

        template <class In, class Filter>
        auto apply_filter(In& in, const Filter& filter, std::uint32_t dim_x, std::uint32_t dim_y,
                          async_handle& handle) -> void
        {
            detail::call_filter_application_kernel(handle, in.get(), filter.get(), dim_x, dim_y,
                                                   in.pitch());
        }

        namespace detail
        {
            auto call_normalization_kernel(async_handle& handle, float* out, const float* in,
                                           std::uint32_t dim_x, std::uint32_t dim_y,
                                           std::size_t out_pitch, std::size_t in_pitch,
                                           std::uint32_t filter_size) -> void;
        }

        template <class In>
        auto normalize(In& in, std::uint32_t filter_size) -> void
        {
            detail::call_normalization_kernel(in.async_handle, in.ptr.get(), in.ptr.get(),
                                              in.width, in.height, in.ptr.pitch(), in.ptr.pitch(),
                                              filter_size);
        }

        template <class In, class Out>
        auto shrink(const In& in, Out& out) -> void
        {
            ddrf::cuda::copy(ddrf::cuda::async, out.ptr, in, out.async_handle,
                             out.width, out.height);
        }

        /*
         * Reconstruction
         * */
        auto set_reconstruction_constants(const reconstruction_constants& rc) -> error_type;
        auto set_roi(const region_of_interest& roi) -> error_type;

        namespace detail
        {
            auto call_backprojection_kernel(async_handle& handle, float* vol_ptr, std::size_t vol_pitch,
                                            std::uint32_t dim_x, std::uint32_t dim_y, std::uint32_t dim_z,
                                            float* p_ptr, std::size_t p_pitch,
                                            std::uint32_t p_dim_x, std::uint32_t p_dim_y,
                                            float sin, float cos, bool enable_roi) -> void;
        }

        template <class Vol, class Proj>
        auto backproject(async_handle& handle, Vol& vol, std::uint32_t dim_x, std::uint32_t dim_y, std::uint32_t dim_z,
                         Proj& p, float sin, float cos, bool enable_roi) -> void
        {
            detail::call_backprojection_kernel(handle, vol.get(), vol.pitch(), dim_x, dim_y, dim_z,
                                               p.ptr.get(), p.ptr.pitch(), p.width, p.height, sin, cos, enable_roi);
        }
    }
}

#endif /* DDAFA_CUDA_BACKEND_H_ */
