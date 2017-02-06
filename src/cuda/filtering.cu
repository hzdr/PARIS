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
#include <cstdint>
#include <numeric>

#include <boost/log/trivial.hpp>

#include <glados/cuda/algorithm.h>
#include <glados/cuda/coordinates.h>
#include <glados/cuda/launch.h>
#include <glados/cuda/memory.h>
#include <glados/cuda/sync_policy.h>
#include <glados/cuda/utility.h>
#include <glados/cufft/plan.h>

#include "backend.h"

namespace paris
{
    namespace cuda
    {
        namespace
        {
            __global__ void filter_creation_kernel(float* __restrict__ r,
                                                   const std::int32_t* __restrict__ j,
                                                   std::uint32_t size, float tau)
            {
                auto x = glados::cuda::coord_x();

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

            __global__ void k_creation_kernel(cufftComplex* __restrict__ data,
                                              std::uint32_t filter_size, float tau)
            {
                auto x = glados::cuda::coord_x();
                if(x < filter_size)
                {
                    auto result = tau * fabsf(sqrtf(powf(data[x].x, 2.f)
                                  + powf(data[x].y, 2.f)));

                    data[x].x = result;
                    data[x].y = result;
                }
            }

            __global__ void filter_application_kernel(cufftComplex* __restrict__ data,
                                                      const cufftComplex* __restrict__ filter,
                                                      std::uint32_t filter_size,
                                                      std::uint32_t data_height,
                                                      std::size_t pitch)
            {
                auto x = glados::cuda::coord_x();
                auto y = glados::cuda::coord_y();

                if((x < filter_size) && (y < data_height))
                {
                    auto row = reinterpret_cast<cufftComplex*>(
                                reinterpret_cast<char*>(data) + y * pitch);

                    row[x].x *= filter[x].x;
                    row[x].y *= filter[x].y;
                }
            }

            __global__ void normalization_kernel(cufftReal* dst, std::size_t pitch,
                                                 std::uint32_t width, std::uint32_t height,
                                                 std::uint32_t filter_size)
            {
                auto x = glados::cuda::coord_x();
                auto y = glados::cuda::coord_y();

                if((x < width) && (y < height))
                {
                    auto row = reinterpret_cast<cufftReal*>(reinterpret_cast<char*>(dst) + y * pitch);

                    row[x] /= filter_size;
                }
            }

            auto make_filter_real(std::uint32_t filter_size, float tau)
            -> glados::cuda::device_ptr<float>
            {
                /*
                 * for a more detailed description see filter_creation_kernel
                 */
                /* create j on the host and fill it with values from
                 * -(filter_size_ - 2) / 2 to filter_size / 2
                 */
                auto h_j = glados::cuda::make_unique_pinned_host<std::int32_t>(filter_size);
                auto size = static_cast<std::int32_t>(filter_size);
                auto j = -(size - 2) / 2;
                std::iota(h_j.get(), h_j.get() + filter_size, j);

                // create j on the device and copy j from the host to the device
                auto d_j = glados::cuda::make_unique_device<std::int32_t>(filter_size);
                glados::cuda::copy(glados::cuda::sync, d_j, h_j, filter_size);

                // create r on the device
                auto d_r  = glados::cuda::make_unique_device<float>(filter_size);

                // calculate the filter values
                glados::cuda::launch(filter_size, filter_creation_kernel,
                                   d_r.get(),
                                   static_cast<const std::int32_t*>(d_j.get()),
                                   filter_size, tau);

                return d_r;
            }

            auto expand(const projection_device_buffer_type& src, std::uint32_t src_dim_x,
                              glados::cuda::pitched_device_ptr<float>& dst, std::uint32_t dst_dim_x,
                              std::uint32_t dim_y, cudaStream_t& stream) -> void
            {
                // reset the expansion destination
                glados::cuda::fill(glados::cuda::async, dst, 0, stream, dst_dim_x, dim_y);

                // copy original projection to expanded projection
                glados::cuda::copy(glados::cuda::async, dst, src, stream, src_dim_x, dim_y);
            }

            auto shrink(const glados::cuda::pitched_device_ptr<float>& src,
                              projection_device_buffer_type& dst, std::uint32_t dim_x, std::uint32_t dim_y,
                              cudaStream_t& stream) -> void
            {
                glados::cuda::copy(glados::cuda::async, dst, src, stream, dim_x, dim_y);
            }
        }

        auto make_filter(std::uint32_t size, float tau) -> filter_buffer_type
        {
            auto r = make_filter_real(size, tau);

            auto size_trans = size / 2 + 1;
            auto k = glados::cuda::make_unique_device<cufftComplex>(size_trans);

            auto n = static_cast<int>(size);

            auto plan = glados::cufft::plan<CUFFT_R2C>{n};
            plan.execute(r.get(), k.get());

            glados::cuda::launch(size_trans, k_creation_kernel, k.get(), size_trans, tau);

            return k;
        }

        auto apply_filter(projection_device_type& p, const filter_buffer_type& k,
                          std::uint32_t filter_size, std::uint32_t n_col)
            -> void
        {
            /* due to cuFFT's crazy API we cannot make the constants which we need as pointers actually const
             * - this applies to n, p_exp_nembed and p_trans_nembed
             */

            // dimensionality of the FFT - 1 in this case
            constexpr auto rank = 1;

            // FFT size for each dimension
            static auto n = static_cast<int>(filter_size);

            // batched FFT -> set batch size
            static const auto batch = static_cast<int>(n_col);

            // allocate memory for expanded projection (projection width -> filter size)
            thread_local static auto p_exp = glados::cuda::make_unique_device<float>(filter_size, n_col);

            // allocate memory for transformed projection
            static const auto size_trans = filter_size / 2 + 1;
            thread_local static auto p_trans = glados::cuda::make_unique_device<cufftComplex>(size_trans, n_col);

            // set distance between the first elements of two successive lines
            static const auto p_exp_dist = static_cast<int>(p_exp.pitch() / sizeof(float));
            static const auto p_trans_dist = static_cast<int>(p_trans.pitch() / sizeof(cufftComplex));

            // set distance between two successive elements
            constexpr auto p_exp_stride = 1;
            constexpr auto p_trans_stride = 1;

            // set storage dimensions of data in memory
            static auto p_exp_nembed = static_cast<int>(p_exp_dist);
            static auto p_trans_nembed = static_cast<int>(p_trans_dist);

            // create plans for forward and inverse FFT
            thread_local static auto forward = glados::cufft::plan<CUFFT_R2C>{rank, &n,
                                                                &p_exp_nembed, p_exp_stride, p_exp_dist,
                                                                &p_trans_nembed, p_trans_stride, p_trans_dist,
                                                                batch};

            thread_local static auto inverse = glados::cufft::plan<CUFFT_C2R>{rank, &n,
                                                                &p_trans_nembed, p_trans_stride, p_trans_dist,
                                                                &p_exp_nembed, p_exp_stride, p_exp_dist,
                                                                batch};

            // assign projection stream to plans
            forward.set_stream(p.meta.stream);
            inverse.set_stream(p.meta.stream);

            // expand and transform the projection
            expand(p.buf, p.dim_x, p_exp, filter_size, n_col, p.meta.stream);
            forward.execute(p_exp.get(), p_trans.get());

            // apply filter to transformed projection
            glados::cuda::launch_async(p.meta.stream, size_trans, n_col,
                                       filter_application_kernel,
                                       p_trans.get(), static_cast<const cufftComplex*>(k.get()),
                                       size_trans, n_col, p_trans.pitch());

            // inverse transformation
            inverse.execute(p_trans.get(), p_exp.get());

            // shrink to original size and normalize
            shrink(p_exp, p.buf, p.dim_x, n_col, p.meta.stream);
            glados::cuda::launch_async(p.meta.stream, p.dim_x, p.dim_y,
                                       normalization_kernel,
                                       p.buf.get(), p.buf.pitch(), p.dim_x, p.dim_y, filter_size);
        }
    }
}
