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

#include <cstddef>
#include <cstdint>

#include <boost/log/trivial.hpp>

#include <glados/cuda/coordinates.h>
#include <glados/cuda/launch.h>

#include "../exception.h"
#include "../reconstruction_constants.h"
#include "../region_of_interest.h"

#include "backend.h"

namespace ddafa
{
    namespace cuda
    {
        namespace
        {
            // note that each device will automatically keep track of its own symbol
            __device__ __constant__ reconstruction_constants dev_consts__{};
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
            __global__ void backproject(float* __restrict__ vol, std::size_t vol_pitch,
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
        }

        auto set_reconstruction_constants(const reconstruction_constants& rc) -> error_type
        {
            return cudaMemcpyToSymbol(dev_consts__, &rc, sizeof(rc));
        }

        auto set_roi(const region_of_interest& roi) -> error_type
        {
            return cudaMemcpyToSymbol(dev_roi__, &roi, sizeof(roi));
        }

        namespace detail
        {
            auto call_backprojection_kernel(async_handle& handle,
                                            float* vol_ptr, std::size_t vol_pitch,
                                            std::uint32_t dim_x, std::uint32_t dim_y,
                                                std::uint32_t dim_z,
                                            float* p_ptr, std::size_t p_pitch,
                                            std::uint32_t p_dim_x, std::uint32_t p_dim_y,
                                            float sin, float cos, bool enable_roi) -> void
            {
                // create a CUDA texture from the projection
                auto res_desc = cudaResourceDesc{};
                res_desc.resType = cudaResourceTypePitch2D;
                res_desc.res.pitch2D.desc = cudaCreateChannelDesc<float>();
                res_desc.res.pitch2D.devPtr = reinterpret_cast<void*>(p_ptr);
                res_desc.res.pitch2D.width = p_dim_x;
                res_desc.res.pitch2D.height = p_dim_y;
                res_desc.res.pitch2D.pitchInBytes = p_pitch;

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
                    throw stage_runtime_error{"reconstruction_stage::process() failed"};
                }

                if(enable_roi)
                    glados::cuda::launch_async(handle, dim_x, dim_y, dim_z, backproject<true>,
                                                vol_ptr, vol_pitch, tex, sin, cos);
                else
                    glados::cuda::launch_async(handle, dim_x, dim_y, dim_z, backproject<false>,
                                                vol_ptr, vol_pitch, tex, sin, cos);

                err = cudaDestroyTextureObject(tex);
                if(err != cudaSuccess)
                {
                    BOOST_LOG_TRIVIAL(fatal) << "Could not destroy CUDA texture: " << cudaGetErrorString(err);
                    throw stage_runtime_error{"reconstruction_stage::process() failed"};
                }
            }
        }
    }
}
