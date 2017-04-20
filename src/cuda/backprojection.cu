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

#include <cstddef>
#include <cstdint>
#include <future>
#include <mutex>
#include <queue>
#include <thread>

#include <boost/log/trivial.hpp>

#include <glados/cuda/coordinates.h>
#include <glados/cuda/launch.h>
#include <glados/cuda/utility.h>

#include "../exception.h"
#include "../region_of_interest.h"

#include "backend.h"
#include "backprojection_constants.h"

namespace paris
{
    namespace cuda
    {
        namespace
        {
            // note that each device will automatically keep track of its own symbol
            __device__ __constant__ backprojection_constants dev_consts__{};
            __device__ __constant__ region_of_interest dev_roi__{};

            inline __device__ auto vol_centered_coordinate(unsigned int coord,
                                                           std::uint32_t dim, float size)
            -> float
            {
                auto size2 = size / 2.f;
                return -(dim * size2) + size2 + coord * size;
            }

            inline __device__ auto proj_real_coordinate(float coord, std::uint32_t dim,
                                                        float size, float offset) -> float
            {
                auto size2 = size / 2.f;
                auto min = -(dim * size2) - offset;
                return (coord - min) / size - (1.f / 2.f);
            }

            template <bool enable_roi>
            __global__ void backprojection_kernel(float* __restrict__ vol, std::size_t vol_pitch,
                                                  cudaTextureObject_t proj, float angle_sin,
                                                  float angle_cos)
            {
                auto k = glados::cuda::coord_x();
                auto l = glados::cuda::coord_y();
                auto m = glados::cuda::coord_z();

                if((k < dev_consts__.vol_dim_x) &&
                   (l < dev_consts__.vol_dim_y) &&
                   (m < dev_consts__.vol_dim_z))
                {
                    auto slice_pitch = vol_pitch * dev_consts__.vol_dim_y;
                    auto slice = reinterpret_cast<char*>(vol) + m * slice_pitch;
                    auto row = reinterpret_cast<float*>(slice + l * vol_pitch);

                    // load old value from global memory while executing other instructions
                    auto old_val = row[k];
                    
                    // add ROI offset. If enable_roi == false, this will be optimized away
                    if(enable_roi)
                    {
                        k += dev_roi__.x1;
                        l += dev_roi__.y1;
                        m += dev_roi__.z1;
                    }

                    // add offset for the current subvolume
                    m += dev_consts__.vol_offset;

                    // get centered coordinates -- volume center at (0, 0, 0)
                    auto x_k = vol_centered_coordinate(k, dev_consts__.vol_dim_x_full,
                                                            dev_consts__.l_vx_x);
                    auto y_l = vol_centered_coordinate(l, dev_consts__.vol_dim_y_full,
                                                            dev_consts__.l_vx_y);
                    auto z_m = vol_centered_coordinate(m, dev_consts__.vol_dim_z_full,
                                                            dev_consts__.l_vx_z);

                    // rotate coordinates
                    auto s = x_k * angle_cos + y_l * angle_sin;
                    auto t = -x_k * angle_sin + y_l * angle_cos;

                    // project rotated coordinates
                    auto factor = dev_consts__.d_sd / (s + dev_consts__.d_so);
                    // add 0.5 to each coordinate to deal with CUDA's filtering mechanism
                    auto h = proj_real_coordinate(t * factor, dev_consts__.proj_dim_x,
                                                                dev_consts__.l_px_x,
                                                                dev_consts__.delta_s) + 0.5f;
                    auto v = proj_real_coordinate(z_m * factor, dev_consts__.proj_dim_y,
                                                                dev_consts__.l_px_y,
                                                                dev_consts__.delta_t) + 0.5f;

                    // get projection value (note the implicit linear interpolation)
                    auto det = tex2D<float>(proj, h, v);

                    // backproject
                    auto u = -(dev_consts__.d_so / (s + dev_consts__.d_so));

                    // restore old coordinate for writing.
                    if(enable_roi)
                        k -= dev_roi__.x1;

                    // write value
                    row[k] = old_val + 0.5f * det * u * u;
                }
            }

            auto do_backprojection(std::mutex& m, std::queue<projection_device_type>& q,
                                   float* v, std::size_t v_pitch, std::uint32_t dim_x, std::uint32_t dim_y,
                                   std::uint32_t dim_z, cudaStream_t v_stream, bool enable_roi,
                                   const region_of_interest& roi, std::promise<void> done_promise, int device) -> void
            {
                if(v == nullptr)
                    return;

                auto&& lock = std::unique_lock<std::mutex>(m, std::defer_lock);
                cudaSetDevice(device);

                while(true)
                {
                    while(q.empty())
                        std::this_thread::yield();

                    // acquire projection
                    lock.lock();
                    auto p = std::move(q.front());
                    q.pop();
                    lock.unlock();

                    if(!p.valid)
                        break;

                    auto sin = std::sin(p.phi);
                    auto cos = std::cos(p.phi);

                    // ensure all previous operations on the projection are complete
                    glados::cuda::synchronize_stream(p.meta.stream);

                    // create a CUDA texture from the projection
                    auto res_desc = cudaResourceDesc{};
                    res_desc.resType = cudaResourceTypePitch2D;
                    res_desc.res.pitch2D.desc = cudaCreateChannelDesc<float>();
                    res_desc.res.pitch2D.devPtr = reinterpret_cast<void*>(p.buf.get());
                    res_desc.res.pitch2D.width = p.dim_x;
                    res_desc.res.pitch2D.height = p.dim_y;
                    res_desc.res.pitch2D.pitchInBytes = p.buf.pitch();

                    auto tex_desc = cudaTextureDesc{};
                    tex_desc.addressMode[0] = cudaAddressModeBorder;
                    tex_desc.addressMode[1] = cudaAddressModeBorder;
                    tex_desc.filterMode = cudaFilterModeLinear;
                    tex_desc.readMode = cudaReadModeElementType;
                    tex_desc.normalizedCoords = 0;

                    auto tex = cudaTextureObject_t{0};
                    auto err = cudaCreateTextureObject(&tex, &res_desc, &tex_desc, nullptr);
                    if(err != cudaSuccess)
                    {
                        BOOST_LOG_TRIVIAL(fatal) << "Could not create CUDA texture: " << cudaGetErrorString(err);
                        throw stage_runtime_error{"backproject() failed"};
                    }

                    auto block_size = dim3{16, 8, 2};
                    auto blocks_x = (dim_x + 16 - (dim_x % 16)) / 16;
                    auto blocks_y = (dim_y + 8 - (dim_y % 8)) / 8;
                    auto blocks_z = (dim_z + 2 - (dim_z % 2)) / 2;
                    auto grid_size = dim3{blocks_x, blocks_y, blocks_z};

                    // apply ROI as needed and backproject
                    if(enable_roi)
                    {
                        err = cudaMemcpyToSymbolAsync(dev_roi__, &roi, sizeof(roi), 0u, cudaMemcpyHostToDevice, v_stream);
                        if(err != cudaSuccess)
                        {
                            BOOST_LOG_TRIVIAL(fatal) << "Could not initialise device ROI: " << cudaGetErrorString(err);
                            throw stage_runtime_error{"backproject() failed"};
                        }

                        /*glados::cuda::launch_async(v_stream, dim_x, dim_y, dim_z, backprojection_kernel<true>,
                                                   v, v_pitch, tex, sin, cos);*/
                        backprojection_kernel<true><<<grid_size, block_size, 0u, v_stream>>>(v, v_pitch, tex, sin, cos);
                    }
                    else
                        backprojection_kernel<false><<<grid_size, block_size, 0u, v_stream>>>(v, v_pitch, tex, sin, cos);
/*                        glados::cuda::launch_async(v_stream, dim_x, dim_y, dim_z, backprojection_kernel<false>,
                                                   v, v_pitch, tex, sin, cos);*/

                    err = cudaDestroyTextureObject(tex);
                    if(err != cudaSuccess)
                    {
                        BOOST_LOG_TRIVIAL(fatal) << "Could not destroy CUDA texture: " << cudaGetErrorString(err);
                        throw stage_runtime_error{"backproject() failed"};
                    }

                    // release projection stream
                    cudaStreamDestroy(p.meta.stream);
                }
                // synchronize backprojection kernel
                // glados::cuda::synchronize_stream(v_stream);
                done_promise.set_value_at_thread_exit(); // notify waiting threads
            }
        }

        auto backproject(projection_device_type& p, volume_device_type& v, std::uint32_t v_offset,
                         const detector_geometry& det_geo, const volume_geometry& vol_geo,
                         bool enable_roi, const region_of_interest& roi,
                         float delta_s, float delta_t)  -> void
        {
            // constants for the backprojection - these never change
            static const auto v_dim_x_full = vol_geo.dim_x;
            static const auto v_dim_y_full = vol_geo.dim_y;
            static const auto v_dim_z_full = vol_geo.dim_z;

            static const auto l_vx_x = vol_geo.l_vx_x;
            static const auto l_vx_y = vol_geo.l_vx_y;
            static const auto l_vx_z = vol_geo.l_vx_z;

            static const auto p_dim_x = det_geo.n_row;
            static const auto p_dim_y = det_geo.n_col;

            static const auto l_px_x = det_geo.l_px_row;
            static const auto l_px_y = det_geo.l_px_col;

            static const auto d_s = delta_s;
            static const auto d_t = delta_t;

            static const auto d_so = det_geo.d_so;
            static const auto d_sd = std::abs(det_geo.d_so) + std::abs(det_geo.d_od);

            // variable for the backprojection - might change between subvolumes
            thread_local static auto offset = std::uint32_t{-1};

            thread_local static auto consts = backprojection_constants{};
            auto consts_changed = (offset != v_offset);
            if(consts_changed)
            {
                offset = v_offset;

                consts = backprojection_constants {
                    v.dim_x,
                    v_dim_x_full,
                    v.dim_y,
                    v_dim_y_full,
                    v.dim_z,
                    v_dim_z_full,
                    offset,
                    l_vx_x,
                    l_vx_y,
                    l_vx_z,
                    p_dim_x,
                    p_dim_y,
                    l_px_x,
                    l_px_y,
                    d_s,
                    d_t,
                    d_so,
                    d_sd
                };

                // initialise device constants
                auto err = cudaMemcpyToSymbolAsync(dev_consts__, &consts, sizeof(consts), 0u, cudaMemcpyHostToDevice,
                                                   v.meta.s.stream);
                if(err != cudaSuccess)
                {
                    BOOST_LOG_TRIVIAL(fatal) << "Could not initialise device constants: " << cudaGetErrorString(err);
                    throw stage_runtime_error{"backproject() failed"};
                }
            }

            thread_local static auto device = int{};
            cudaGetDevice(&device);

            thread_local static auto p_queue = std::queue<projection_device_type>{};
            thread_local static auto&& m = std::mutex{};
            thread_local static auto&& lock = std::unique_lock<std::mutex>{m, std::defer_lock};

            thread_local static auto old_ptr = static_cast<float*>(nullptr);
            if(old_ptr != v.buf.get()) // are we still operating on the same volume?
            {
                old_ptr = v.buf.get();
                auto&& done_promise = std::promise<void>{};
                v.meta.done_future = done_promise.get_future();

                auto bp_worker = std::thread{do_backprojection, std::ref(m), std::ref(p_queue),
                                                             v.buf.get(), v.buf.pitch(), v.dim_x, v.dim_y, v.dim_z,
                                                             v.meta.s.stream, enable_roi, std::ref(roi),
                                                             std::move(done_promise), device};
                bp_worker.detach();
            }

            lock.lock();
            p_queue.push(std::move(p));
            lock.unlock();
        }
    }
}
