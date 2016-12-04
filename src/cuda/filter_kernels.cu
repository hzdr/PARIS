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

#include <cstdint>
#include <numeric>

#include <boost/log/trivial.hpp>

#include <ddrf/cuda/algorithm.h>
#include <ddrf/cuda/coordinates.h>
#include <ddrf/cuda/launch.h>
#include <ddrf/cuda/memory.h>
#include <ddrf/cufft/plan.h>

#include "backend.h"

namespace ddafa
{
    namespace cuda
    {
        namespace
        {
            __global__ void filter_creation_kernel(float* __restrict__ r,
                                                   const std::int32_t* __restrict__ j,
                                                   std::uint32_t size, float tau)
            {
                auto x = ddrf::cuda::coord_x();

                /*
                 * r(j) with j = [ -(filter_length - 2) / 2, ..., 0, ..., filter_length / 2 ]
                 * tau = horizontal pixel distance
                 *
                 *          1/8 * 1/(tau^2)                     for j = 0
                 * r(j) = { 0                                   for even j
                 *          -(1 / (2 * j^2 * pi^2 * tau^2))     for odd j
                 */
                if(x < size)
                {
                    if(j[x] == 0) // does j = 0?
                        r[x] = (1.f / 8.f) * (1.f / powf(tau, 2.f));
                    else // j != 0
                    {
                        if(j[x] % 2 == 0) // is j even?
                            r[x] = 0.f;
                        else // j is odd
                            r[x] = -(1.f / (2.f * powf(j[x], 2.f)
                                   * powf(M_PI, 2.f)
                                   * powf(tau, 2.f)));
                    }
                }
            } 

            auto make_filter_real(std::uint32_t filter_size, float tau)
            -> ddrf::cuda::device_ptr<float>
            {
                /*
                 * for a more detailed description see filter_creation_kernel
                 */
                /* create j on the host and fill it with values from
                 * -(filter_size_ - 2) / 2 to filter_size / 2
                 */
                auto h_j = ddrf::cuda::make_unique_pinned_host<std::int32_t>(filter_size);
                auto size = static_cast<std::int32_t>(filter_size);
                auto j = -(size - 2) / 2;
                std::iota(h_j.get(), h_j.get() + filter_size, j);
                BOOST_LOG_TRIVIAL(debug) << "Host filter creation succeeded";

                // create j on the device and copy j from the host to the device
                auto d_j = ddrf::cuda::make_unique_device<std::int32_t>(filter_size);
                ddrf::cuda::copy(ddrf::cuda::sync, d_j, h_j, filter_size);
                BOOST_LOG_TRIVIAL(debug) << "Copied filter from host to device";

                // create r on the device
                auto d_r  = ddrf::cuda::make_unique_device<float>(filter_size);

                // calculate the filter values
                ddrf::cuda::launch(filter_size, filter_creation_kernel,
                                   d_r.get(),
                                   static_cast<const std::int32_t*>(d_j.get()),
                                   filter_size, tau);
                BOOST_LOG_TRIVIAL(debug) << "Device filter creation succeeded";

                BOOST_LOG_TRIVIAL(debug) << "Filter creation complete";
                return d_r;
            }

            __global__ void k_creation_kernel(cufftComplex* __restrict__ data,
                                              std::uint32_t filter_size, float tau)
            {
                auto x = ddrf::cuda::coord_x();
                if(x < filter_size)
                {
                    auto result = tau * fabsf(sqrtf(powf(data[x].x, 2.f)
                                  + powf(data[x].y, 2.f)));

                    data[x].x = result;
                    data[x].y = result;
                }
            }
        }

        auto make_filter(std::uint32_t size, float tau) -> device_ptr_1D<fft::complex_type>
        {
            auto r = make_filter_real(size, tau);

            auto size_trans = size / 2 + 1;
            auto k = ddrf::cuda::make_unique_device<cufftComplex>(size_trans);

            auto n = static_cast<int>(size);

            auto plan = ddrf::cufft::plan<CUFFT_R2C>{n};
            plan.execute(r.get(), k.get());

            ddrf::cuda::launch(size_trans, k_creation_kernel, k.get(), size_trans, tau);

            return k;
        }

        namespace
        {
            __global__ void filter_application_kernel(cufftComplex* __restrict__ data,
                                                      const cufftComplex* __restrict__ filter,
                                                      std::uint32_t filter_size,
                                                      std::uint32_t data_height,
                                                      std::size_t pitch)
            {
                auto x = ddrf::cuda::coord_x();
                auto y = ddrf::cuda::coord_y();

                if((x < filter_size) && (y < data_height))
                {
                    auto row = reinterpret_cast<cufftComplex*>(
                                reinterpret_cast<char*>(data) + y * pitch);

                    row[x].x *= filter[x].x;
                    row[x].y *= filter[x].y;
                }
            }
        }

        namespace detail
        {
            auto call_filter_application_kernel(async_handle& handle, fft::complex_type* in,
                                                const fft::complex_type* filter,
                                                std::uint32_t dim_x, std::uint32_t dim_y,
                                                std::size_t in_pitch) -> void
            {
                ddrf::cuda::launch_async(handle, dim_x, dim_y, filter_application_kernel,
                                         in, filter, dim_x, dim_y, in_pitch);
            }
        }

        namespace
        {
            __global__ void normalization_kernel(cufftReal* dst, std::size_t dst_pitch,
                                                 const cufftReal* src, std::size_t src_pitch,
                                                 std::uint32_t width, std::uint32_t height,
                                                 std::uint32_t filter_size)
            {
                auto x = ddrf::cuda::coord_x();
                auto y = ddrf::cuda::coord_y();

                if((x < width) && (y < height))
                {
                    auto dst_row = reinterpret_cast<cufftReal*>(
                                    reinterpret_cast<char*>(dst) + y * dst_pitch);
                    auto src_row = reinterpret_cast<const cufftReal*>(
                                    reinterpret_cast<const char*>(src) + y * src_pitch);

                    dst_row[x] = src_row[x] / filter_size;
                }
            }
        }

        namespace detail
        {
            auto call_normalization_kernel(async_handle& handle, float* out, const float* in,
                                           std::uint32_t dim_x, std::uint32_t dim_y,
                                           std::size_t out_pitch, std::size_t in_pitch,
                                           std::uint32_t filter_size) -> void
            {
                ddrf::cuda::launch_async(handle, dim_x, dim_y, normalization_kernel,
                                            out, out_pitch, in, in_pitch, dim_x, dim_y,
                                            filter_size);
            }
        }
    }
}
