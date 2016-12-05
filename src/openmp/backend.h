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
 * Date: 04 December 2016
 * Authors: Jan Stephan <j.stephan@hzdr.de>
 */

#ifndef DDAFA_OPENMP_BACKEND_H_
#define DDAFA_OPENMP_BACKEND_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <fftw3.h>

#include "../reconstruction_constants.h"
#include "../region_of_interest.h"
#include "../subvolume_information.h"

namespace ddafa
{
    namespace openmp
    {
        /*
         * Runtime management
         * */
        using error_type = int;
        constexpr auto success = 1;
        auto print_error(const std::string& msg, error_type err) noexcept -> void;

        // exceptions
        using bad_alloc = std::bad_alloc;
        using invalid_argument = std::invalid_argument;
        using runtime_error = std::runtime_error;

        constexpr auto name = "OpenMP";

        /*
         * Memory management
         * */
        using allocator = void; // FIXME

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

        constexpr auto set_device(const device_handle&) {}
        constexpr auto set_device_noexcept(const device_handle&) noexcept { return success; }

        auto get_devices() { return std::vector<device_handle>{0}; }

        /*
         * Synchronization
         * */
        constexpr auto sync = int{};
        constexpr auto async = int{};

        using async_handle = int;
        constexpr auto default_async_handle = 0;

        auto make_async_handle() { return async_handle{0}; }
        auto destroy_async_handle(async_handle&) noexcept { return success; }

        auto synchronize(const async_handle&) {}

        /*
         * Basic algorithms
         * */
        template <class SyncPolicy, class D, class S, class... Args>
        auto copy(SyncPolicy&& /* policy */ , D& dst, const S& src, Args&&... args) -> void
        {
        }

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

            // FIXME
            using bad_alloc = void;
            using invalid_argument = void;
            using runtime_error = void;

            using complex_type = fftwf_complex;
            using forward_plan_type = fftwf_plan;
            using inverse_plan_type = fftwf_plan;

            auto make_forward_plan(int rank, int* n, int batch_size,
                                float* in, int* inembed, int istride, int idist,
                                complex_type* out, int* onembed, int ostride, int odist) -> forward_plan_type;

            auto make_inverse_plan(int rank, int* n, int batch_size,
                                complex_type* in, int* inembed, int istride, int idist,
                                float* out, int* onembed, int ostride, int odist) -> inverse_plan_type;
        }

        auto make_filter(std::uint32_t size,float tau) -> device_ptr_1D<fft::complex_type>;

        template <class Ptr>
        auto calculate_distance(Ptr& /* p */, std::uint32_t width) -> int
        {
            return width;
        }

        template <class In, class Out>
        auto expand(const In& in, Out& out, std::uint32_t dim_x, std::uint32_t dim_y) -> void
        {
            auto in_ptr = in.ptr.get();
            auto out_ptr = out.get();

            // reset expanded projection
            std::fill_n(out_ptr, dim_x, 0.f);

            // copy original projection to expanded projection
            for(auto y = 0u; y < dim_y; ++y)
                std::copy_n(in_ptr, in.width, out_ptr);
        }

        template <class In, class Out, class Plan>
        auto transform(const In& /* in */, Out& /* out */, Plan& plan, async_handle&) -> void
        {
            fftw_execute(plan);
        }
    }
}

#endif

