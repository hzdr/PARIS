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

#ifndef PARIS_OPENMP_BACKEND_H_
#define PARIS_OPENMP_BACKEND_H_

#include <algorithm>
#include <cmath> // remove me
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <boost/log/trivial.hpp> // remove me

#include <fftw3.h>

#include <glados/generic/allocator.h>

#include "../reconstruction_constants.h"
#include "../region_of_interest.h"
#include "../subvolume_information.h"

namespace paris
{
    namespace openmp
    {
        /*
         * Runtime management
         * */
        using error_type = int;
        constexpr auto success = 1;
        inline auto print_error(const std::string& msg, error_type err) noexcept -> void {}

        // exceptions
        using bad_alloc = std::bad_alloc;
        using invalid_argument = std::invalid_argument;
        using runtime_error = std::runtime_error;

        constexpr auto name = "OpenMP";

        /*
         * Memory management
         * */
        using allocator = glados::generic::allocator<float, glados::memory_layout::pointer_2D>;

        template <class T>
        using host_ptr_1D = std::unique_ptr<T[]>;

        template <class T>
        using host_ptr_2D = std::unique_ptr<T[]>;

        template <class T>
        using host_ptr_3D = std::unique_ptr<T[]>;

        template <class T>
        using device_ptr_1D = std::unique_ptr<T[]>;

        template <class T>
        using device_ptr_2D = std::unique_ptr<T[]>;

        template <class T>
        using device_ptr_3D = std::unique_ptr<T[]>;

        template <class T>
        auto make_host_ptr(std::size_t n) -> host_ptr_1D<T>
        {
            return std::make_unique<T[]>(n);
        }

        template <class T>
        auto make_host_ptr(std::size_t x, std::size_t y) -> host_ptr_2D<T>
        {
            return std::make_unique<T[]>(x * y);
        }

        template <class T>
        auto make_host_ptr(std::size_t x, std::size_t y, std::size_t z) -> host_ptr_3D<T>
        {
            return std::make_unique<T[]>(x * y * z);
        }

        template <class T>
        auto make_device_ptr(std::size_t n) -> device_ptr_1D<T>
        {
            return std::make_unique<T[]>(n);
        }

        template <class T>
        auto make_device_ptr(std::size_t x, std::size_t y) -> device_ptr_2D<T>
        {
            return std::make_unique<T[]>(x * y);
        }

        template <class T>
        auto make_device_ptr(std::size_t x, std::size_t y, std::size_t z) -> device_ptr_3D<T>
        {
            return std::make_unique<T[]>(x * y * z);
        }

        /*
         * Device management
         * */
        using device_handle = int;

        constexpr auto set_device(const device_handle&) { return 0; } // dummy return to satisfy icc
        constexpr auto set_device_noexcept(const device_handle&) noexcept { return success; }

        inline auto get_devices() { return std::vector<device_handle>{0}; }

        /*
         * Synchronization
         * */
        constexpr auto sync = int{};
        constexpr auto async = int{};

        using async_handle = int;
        constexpr auto default_async_handle = async_handle{0};

        inline auto make_async_handle() { return async_handle{0}; }
        inline auto destroy_async_handle(async_handle&) noexcept { return success; }

        inline auto synchronize(const async_handle&) {}

        /*
         * Basic algorithms
         * */
        namespace detail
        {
            template <class D, class S>
            auto copy(D& dst, const S& src, std::size_t n) noexcept -> void
            {
                std::copy_n(src.get(), n, dst.get());
            }

            template <class D, class S>
            auto copy(D& dst, const S& src, std::size_t x, std::size_t y) noexcept -> void
            {
                std::copy_n(src.get(), x * y, dst.get());
            }

            template <class D, class S>
            auto copy(D& dst, const S& src, std::size_t x, std::size_t y, std::size_t z) noexcept -> void
            {
                std::copy_n(src.get(), x * y * z, dst.get());
            }

            template <class D>
            auto fill(D& dst, int value, std::size_t n) noexcept -> void
            {
                auto bytes = sizeof(typename D::element_type) * n;
                auto ptr = reinterpret_cast<char*>(dst.get());
                std::fill_n(ptr, bytes, value);
            }

            template <class D>
            auto fill(D& dst, int value, std::size_t x, std::size_t y) noexcept -> void
            {
                auto bytes = sizeof(typename D::element_type) * x * y;
                auto ptr = reinterpret_cast<char*>(dst.get());
                std::fill_n(ptr, bytes, value);
            }

            template <class D>
            auto fill(D& dst, int value, std::size_t x, std::size_t y, std::size_t z) noexcept -> void
            {
                auto bytes = sizeof(typename D::element_type) * x * y * z;
                auto ptr = reinterpret_cast<char*>(dst.get());
                std::fill_n(ptr, bytes, value);
            }
        }

        template <class SyncPolicy, class D, class S, class... Args>
        auto copy(SyncPolicy&& /* policy */ , D& dst, const S& src, Args&&... args) -> void
        {
            detail::copy(dst, src, std::forward<Args>(args)...);
        }

        // remove async_handle
        template <class SyncPolicy, class D, class S, class... Args>
        auto copy(SyncPolicy&& /* policy */, D& dst, const S& src, async_handle& /* handle */, Args&&... args) -> void
        {
            detail::copy(dst, src, std::forward<Args>(args)...);
        }

        template <class SyncPolicy, class D, class... Args>
        auto fill(SyncPolicy&& /* policy */, D& dst, int value, Args&&... args) -> void
        {
            detail::fill(dst, value, std::forward<Args>(args)...);
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
        auto make_subvolume_information(const volume_geometry& vol_geo, const detector_geometry& det_geo, int proj_num) noexcept -> subvolume_info;

        /*
         * Weighting
         * */
        namespace detail
        {
            auto do_weighting(float* out, const float* in,  std::uint32_t dim_x, std::uint32_t dim_y,
                              float h_min, float v_min, float d_sd, float l_px_row, float l_px_col) noexcept -> void;
        }

        template <class Proj>
        auto weight(Proj& p, float h_min, float v_min, float d_sd, float l_px_row, float l_px_col) -> void
        {
            detail::do_weighting(p.ptr.get(), p.ptr.get(),
                                 p.width, p.height, h_min, v_min, d_sd, l_px_row, l_px_col);
        }

        /*
         * Filtering
         * */
        namespace fft
        {
            constexpr auto name = "FFTW";

            // FIXME -- we should throw our own exceptions
            using bad_alloc = std::bad_alloc;
            using invalid_argument = std::invalid_argument;
            using runtime_error = std::runtime_error;

            using real_type = float;
            using complex_type = fftwf_complex;
            using forward_plan_type = fftwf_plan;
            using inverse_plan_type = fftwf_plan;

            auto make_forward_plan(int rank, int* n, int batch_size,
                                float* in, int* inembed, int istride, int idist,
                                complex_type* out, int* onembed, int ostride, int odist) -> forward_plan_type;

            auto make_inverse_plan(int rank, int* n, int batch_size,
                                complex_type* in, int* inembed, int istride, int idist,
                                float* out, int* onembed, int ostride, int odist) -> inverse_plan_type;

            struct deleter
            {
                auto operator()(void* p) noexcept -> void
                {
                    fftwf_free(p);
                }
            };

            template <class T>
            using pointer = std::unique_ptr<T[], deleter>;

            template <class T>
            auto make_ptr(std::size_t n) -> pointer<T>
            {
                auto p = reinterpret_cast<T*>(fftwf_malloc(n * sizeof(T)));
                return pointer<T>{p};
            }

            template <class T>
            auto make_ptr(std::size_t x, std::size_t y) -> pointer<T>
            {
                auto p = reinterpret_cast<T*>(fftwf_malloc(x * y  * sizeof(T)));
                return pointer<T>{p};
            }
        }

        auto make_filter(std::uint32_t size, float tau) -> fft::pointer<fft::complex_type>;

        template <class Ptr>
        auto calculate_distance(Ptr& /* p */, std::uint32_t width) -> int
        {
            return static_cast<int>(width);
        }

        template <class In, class Out>
        auto expand(const In& in, Out& out, std::uint32_t dim_x, std::uint32_t dim_y) -> void
        {
            auto in_ptr = in.ptr.get();
            auto out_ptr = out.get();

            // copy original projection to expanded projection
            #pragma omp parallel for collapse(2)
            for(auto y = 0u; y < in.height; ++y)
            {
                for(auto x = 0u; x < dim_x; ++x)
                {
                    if(x < in.width)
                        out_ptr[x + y * dim_x] = in_ptr[x + y * in.width];
                    else
                        out_ptr[x + y * dim_x] = 0;
                }
            }
        }

        template <class In, class Out, class Plan>
        auto transform(const In& /* in */, Out& /* out */, Plan& plan, async_handle&) -> void
        {
            fftwf_execute(plan);
        }

        namespace detail
        {
            auto do_filtering(fft::complex_type* in, const fft::complex_type* filter,
                              std::uint32_t dim_x, std::uint32_t dim_y) -> void;
        }

        template <class In, class Filter>
        auto apply_filter(In& in, const Filter& filter, std::uint32_t dim_x, std::uint32_t dim_y,
                            async_handle& /* handle */) -> void
        {
            detail::do_filtering(in.get(), filter.get(), dim_x, dim_y);
        }

        template <class In, class Out>
        auto shrink(const In& in, Out& out, std::uint32_t filter_size) -> void
        {
            auto in_ptr = in.get();
            auto out_ptr = out.ptr.get();
            // copy expanded projection to original projection
            #pragma omp parallel for collapse(2)
            for(auto y = 0u; y < out.height; ++y)
            {
                for(auto x = 0u; x < out.width; ++x)
                {
                    auto coord = x + y * out.width;
                    out_ptr[x + y * out.width] = in_ptr[x + y * filter_size];
                }
            }
        }

        template <class In>
        auto normalize(In& in, std::uint32_t filter_size) -> void
        {
            auto in_ptr = in.ptr.get();

            #pragma omp parallel for collapse(2)
            for(auto y = 0; y < in.height; ++y)
            {
                for(auto x = 0; x < in.width; ++x)
                {
                    in_ptr[x + y * in.width] /= filter_size;
                }
            }
        }

        /*
         * Reconstruction
         * */
        auto set_reconstruction_constants(const reconstruction_constants& rc) noexcept -> error_type;
        auto set_roi(const region_of_interest& roi) noexcept -> error_type;


        namespace detail
        {
            auto do_backprojection(float* vol_ptr, std::uint32_t dim_x, std::uint32_t dim_y, std::uint32_t dim_z,
                                   const float* p_ptr, std::uint32_t p_dim_x, std::uint32_t p_dim_y,
                                   float sin, float cos, bool enable_roi) noexcept -> void;
        }

        template <class Vol, class Proj>
        auto backproject(async_handle& /* handle */,
                         Vol& vol, std::uint32_t dim_x, std::uint32_t dim_y, std::uint32_t dim_z,
                         Proj& p, float sin, float cos, bool enable_roi) noexcept -> void
        {
            detail::do_backprojection(vol.get(), dim_x, dim_y, dim_z, p.ptr.get(), p.width, p.height, sin, cos, enable_roi);
        }
    }
}

#endif
