/*
 * This file is part of the ddafa reconstruction program.
 *
 * Copyright (C) 2016 Helmholtz-Zentrum Dresden-Rossendorf
 *
 * Licensed under the EUPL, Version 1.1 or - as soon they will be approved by
 * the European Commission - subsequent version of the EUPL (the "Licence");
 * You may not use this work except in compliance with the Licence.
 * You may obtain a copy of the Licence at:
 *
 * http://ec.europa.eu/idabc/eupl
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the Licence is distributed on an "AS IS" basis,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the Licence for the specific language governing permissions and
 * limitations under the Licence.
 *
 * Date: 18 August 2016
 * Authors: Jan Stephan
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <utility>

#include <boost/log/trivial.hpp>

#include <ddrf/cuda/algorithm.h>
#include <ddrf/cuda/coordinates.h>
#include <ddrf/cuda/launch.h>
#include <ddrf/cuda/memory.h>
#include <ddrf/cuda/sync_policy.h>
#include <ddrf/cuda/utility.h>

#include "exception.h"
#include "geometry.h"
#include "projection.h"
#include "reconstruction_stage.h"
#include "region_of_interest.h"
#include "scheduler.h"
#include "volume.h"

namespace ddafa
{
    namespace
    {
        // constants for the current subvolume -- these never change between kernel executions
        struct reconstruction_constants
        {
            std::uint32_t vol_dim_x;
            std::uint32_t vol_dim_x_full;
            std::uint32_t vol_dim_y;
            std::uint32_t vol_dim_y_full;
            std::uint32_t vol_dim_z;
            std::uint32_t vol_dim_z_full;
            std::uint32_t vol_offset;

            float l_vx_x;
            float l_vx_y;
            float l_vx_z;

            std::uint32_t proj_dim_x;
            std::uint32_t proj_dim_y;

            float l_px_x;
            float l_px_y;
            
            float delta_s;
            float delta_t;

            float d_so;
            float d_sd;
        };

        // note that each device will automatically keep track of its own symbol, no further synchronization needed
        __device__ __constant__ reconstruction_constants dev_consts__{};
        __device__ __constant__ region_of_interest dev_roi__{};

        inline __device__ auto vol_centered_coordinate(unsigned int coord, std::uint32_t dim, float size) -> float
        {
            auto size2 = size / 2.f;
            return -(dim * size2) + size2 + coord * size;
        }

        inline __device__ auto proj_real_coordinate(float coord, std::uint32_t dim, float size, float offset) -> float
        {
            auto size2 = size / 2.f;
            auto min = -(dim * size2) - offset;
            return (coord - min) / size - (1.f / 2.f);
        }

        template <bool enable_roi>
        __global__ void backproject(float* __restrict__ vol, std::uint32_t vol_w, std::uint32_t vol_h, std::uint32_t vol_d, std::size_t vol_pitch,
                                    std::uint32_t vol_offset, std::uint32_t vol_d_full, float voxel_size_x, float voxel_size_y, float voxel_size_z,
                                    cudaTextureObject_t proj, std::uint32_t proj_w, std::uint32_t proj_h,
                                    float pixel_size_x, float pixel_size_y, float pixel_offset_x, float pixel_offset_y,
                                    float angle_sin, float angle_cos, float dist_src, float dist_sd)
        {
            auto k = ddrf::cuda::coord_x();
            auto l = ddrf::cuda::coord_y();
            auto m = ddrf::cuda::coord_z();

            if((k < vol_w) && (l < vol_h) && (m < vol_d))
            {
                auto slice_pitch = vol_pitch * vol_h;
                auto slice = reinterpret_cast<char*>(vol) + m * slice_pitch;
                auto row = reinterpret_cast<float*>(slice + l * vol_pitch);

                // optimization hackery: load value from global memory while executing other instructions
                auto old_val = row[k];
                
                // add ROI offset. If enable_roi == false, the compiler will optimize this code away
                if(enable_roi)
                {
                    k += dev_roi__.x1;
                    l += dev_roi__.y1;
                    m += dev_roi__.z1;
                }

                // add offset for the current subvolume
                m += vol_offset;

                // get centered coordinates -- volume center is at (0, 0, 0) and the top slice is at -(vol_d_off / 2)
                auto x_k = vol_centered_coordinate(k, vol_w, voxel_size_x);
                auto y_l = vol_centered_coordinate(l, vol_h, voxel_size_y);
                auto z_m = vol_centered_coordinate(m, vol_d_full, voxel_size_z);

                // rotate coordinates
                auto s = x_k * angle_cos + y_l * angle_sin;
                auto t = -x_k * angle_sin + y_l * angle_cos;

                // project rotated coordinates
                auto factor = dist_sd / (s + dist_src);
                // add 0.5 to each coordinate to deal with CUDA's filtering mechanism
                auto h = proj_real_coordinate(t * factor, proj_w, pixel_size_x, pixel_offset_x) + 0.5f;
                auto v = proj_real_coordinate(z_m * factor, proj_h, pixel_size_y, pixel_offset_y) + 0.5f;

                // get projection value (note the implicit linear interpolation)
                auto det = tex2D<float>(proj, h, v);

                // backproject
                auto u = -(dist_src / (s + dist_src));
                row[k] = old_val + 0.5f * det * u * u;
            }
        }

        auto download(const ddrf::cuda::pitched_device_ptr<float>& in, ddrf::cuda::pinned_host_ptr<float>& out,
                        std::uint32_t x, std::uint32_t y, std::uint32_t z) -> void
        {
            ddrf::cuda::copy(ddrf::cuda::sync, out, in, x, y, z);
        }
    }

    reconstruction_stage::reconstruction_stage(int device) noexcept
    : device_{device}
    {
    }

    auto reconstruction_stage::assign_task(task t) noexcept -> void
    {
        det_geo_ = t.det_geo;
        vol_geo_ = t.vol_geo;
        subvol_geo_ = t.subvol_geo;
        enable_angles_ = t.enable_angles;

        enable_roi_ = t.enable_roi;
        roi_ = t.roi;

        task_id_ = t.id;
    }

    auto reconstruction_stage::run() -> void
    {
        auto sre = stage_runtime_error{"reconstruction_stage::run() failed"};

        try
        {
            ddrf::cuda::set_device(device_);

            auto dim_z = std::uint32_t{};
            // if this is the lowest subvolume we need to consider the remaining slices
            if(task_id_ == task_num_ - 1)
                dim_z = subvol_geo_.dim_z + subvol_geo_.remainder;
            else
                dim_z = subvol_geo_.dim_z;

            // create host volume
            auto vol_h_ptr = ddrf::cuda::make_unique_pinned_host<float>(subvol_geo_.dim_x, subvol_geo_.dim_y, dim_z);
            ddrf::cuda::fill(ddrf::cuda::sync, vol_h_ptr, 0, subvol_geo_.dim_x, subvol_geo_.dim_y, dim_z);

            // create device volume
            auto vol_d_ptr = ddrf::cuda::make_unique_device<float>(subvol_geo_.dim_x, subvol_geo_.dim_y, dim_z);
            ddrf::cuda::fill(ddrf::cuda::sync, vol_d_ptr, 0, subvol_geo_.dim_x, subvol_geo_.dim_y, dim_z);

            // calculate offset for the current subvolume
            auto offset = task_id_ * subvol_geo_.dim_z;

            // utility variables
            auto delta_s = det_geo_.delta_s * det_geo_.l_px_row;
            auto delta_t = det_geo_.delta_t * det_geo_.l_px_col;

            // initialize dev_consts__


            // initialize dev_roi__
            if(enable_roi_)
            {
                auto err = cudaMemcpyToSymbol(dev_roi__, &roi_, sizeof(roi_));
                if(err != cudaSuccess)
                {
                    BOOST_LOG_TRIVIAL(fatal) << "Could not initialize region of interest on device: " << cudaGetErrorString(err);
                    throw stage_runtime_error{"reconstruction_stage::process() failed"};
                }
            }

            while(true)
            {
                auto p = input_();
                if(!p.valid)
                    break;

                if(p.idx % 10 == 0)
                    BOOST_LOG_TRIVIAL(info) << "Reconstruction processing projection #" << p.idx << " on device #" << device_ << " in stream " << p.stream;

                // get angular position of the current projection
                auto phi = 0.f;
                if(enable_angles_)
                    phi = p.phi;
                else
                    phi = static_cast<float>(p.idx) * det_geo_.delta_phi;

                // transform to radians
                phi *= static_cast<float>(M_PI) / 180.f;

                auto sin = std::sin(phi);
                auto cos = std::cos(phi);

                // create a CUDA texture from the projection
                auto res_desc = cudaResourceDesc{};
                res_desc.resType = cudaResourceTypePitch2D;
                res_desc.res.pitch2D.desc = cudaCreateChannelDesc<float>();
                res_desc.res.pitch2D.devPtr = reinterpret_cast<void*>(p.ptr.get());
                res_desc.res.pitch2D.width = p.width;
                res_desc.res.pitch2D.height = p.height;
                res_desc.res.pitch2D.pitchInBytes = p.ptr.pitch();

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

                if(enable_roi_)
                    ddrf::cuda::launch_async(p.stream, subvol_geo_.dim_x, subvol_geo_.dim_y, subvol_geo_.dim_z,
                                        backproject<true>,
                                        vol_d_ptr.get(), subvol_geo_.dim_x, subvol_geo_.dim_y, subvol_geo_.dim_z, vol_d_ptr.pitch(),
                                        offset, vol_geo_.dim_z,
                                        vol_geo_.l_vx_x, vol_geo_.l_vx_y, vol_geo_.l_vx_z,
                                        tex, p.width, p.height, det_geo_.l_px_row, det_geo_.l_px_col,
                                            delta_s, delta_t,
                                        sin, cos, det_geo_.d_so, std::abs(det_geo_.d_so) + std::abs(det_geo_.d_od));
                else
                    ddrf::cuda::launch_async(p.stream, subvol_geo_.dim_x, subvol_geo_.dim_y, subvol_geo_.dim_z,
                                        backproject<false>,
                                        vol_d_ptr.get(), subvol_geo_.dim_x, subvol_geo_.dim_y, subvol_geo_.dim_z, vol_d_ptr.pitch(),
                                        offset, vol_geo_.dim_z,
                                        vol_geo_.l_vx_x, vol_geo_.l_vx_y, vol_geo_.l_vx_z,
                                        tex, p.width, p.height, det_geo_.l_px_row, det_geo_.l_px_col,
                                            delta_s, delta_t,
                                        sin, cos, det_geo_.d_so, std::abs(det_geo_.d_so) + std::abs(det_geo_.d_od));


                ddrf::cuda::synchronize_stream(p.stream);

                err = cudaDestroyTextureObject(tex);
                if(err != cudaSuccess)
                {
                    BOOST_LOG_TRIVIAL(fatal) << "Could not destroy CUDA texture: " << cudaGetErrorString(err);
                    throw stage_runtime_error{"reconstruction_stage::process() failed"};
                }
            }

            // copy results to host
            download(vol_d_ptr, vol_h_ptr, subvol_geo_.dim_x, subvol_geo_.dim_y, subvol_geo_.dim_z);

            // create and move output volume -- done
            output_(output_type{std::move(vol_h_ptr), subvol_geo_.dim_x, subvol_geo_.dim_y, dim_z, offset , true, device_});
            BOOST_LOG_TRIVIAL(info) << "Completed task #" << task_id_ << " on device #" << device_;
        }
        catch(const ddrf::cuda::bad_alloc& ba)
        {
            BOOST_LOG_TRIVIAL(fatal) << "reconstruction_stage::run() encountered a bad_alloc: " << ba.what();
            throw sre;
        }
        catch(const ddrf::cuda::invalid_argument& ia)
        {
            BOOST_LOG_TRIVIAL(fatal) << "reconstruction_stage::run() passed an invalid argument to the CUDA runtime: " << ia.what();
            throw sre;
        }
        catch(const ddrf::cuda::runtime_error& re)
        {
            BOOST_LOG_TRIVIAL(fatal) << "reconstruction-stage::run() encountered a CUDA runtime error: " << re.what();
            throw sre;
        }
    }

    auto reconstruction_stage::set_input_function(std::function<input_type(void)> input) noexcept -> void
    {
        input_ = input;
    }

    auto reconstruction_stage::set_output_function(std::function<void(output_type)> output) noexcept -> void
    {
        output_ = output;
    }
}
