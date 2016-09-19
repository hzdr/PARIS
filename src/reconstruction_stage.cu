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
#include <functional>
#include <future>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

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
#include "volume.h"

namespace ddafa
{
    namespace
    {
        std::mutex mutex__;

        inline __device__ auto vol_centered_coordinate(unsigned int coord, std::size_t dim, float size) -> float
        {
            auto size2 = size / 2.f;
            return -(dim * size2) + size2 + coord * size;
        }

        // round and cast as needed
        inline __device__ auto proj_real_coordinate(float coord, std::size_t dim, float size, float offset) -> float
        {
            auto size2 = size / 2.f;
            auto min = -(dim * size2) - offset;
            return (coord - min) / size - (1.f / 2.f);
        }

        template <class T>
        inline __device__ auto as_unsigned(T x) -> unsigned int
        {
            return static_cast<unsigned int>(x);
        }

        __device__ auto interpolate(float h, float v, cudaTextureObject_t proj, std::size_t proj_width, std::size_t proj_height, std::size_t proj_pitch,
                                    float pixel_size_x, float pixel_size_y, float offset_x, float offset_y)
        -> float
        {
            auto h_real = proj_real_coordinate(h, proj_width, pixel_size_x, offset_x);
            auto v_real = proj_real_coordinate(v, proj_height, pixel_size_y, offset_y);

            auto h_j0 = floorf(h_real);
            auto h_j1 = h_j0 + 1.f;
            auto v_i0 = floorf(v_real);
            auto v_i1 = v_i0 + 1.f;

            auto w_h0 = h_real - h_j0;
            auto w_v0 = v_real - v_i0;

            auto w_h1 = 1.f - w_h0;
            auto w_v1 = 1.f - w_v0;

            auto h_j0_ui = as_unsigned(h_j0);
            auto h_j1_ui = as_unsigned(h_j1);
            auto v_i0_ui = as_unsigned(v_i0);
            auto v_i1_ui = as_unsigned(v_i1);

            // ui coordinates might be invalid due to negative v_i0, thus
            // bounds checking
            auto h_j0_valid = (h_j0 >= 0.f);
            auto h_j1_valid = (h_j1 < static_cast<float>(proj_width));
            auto v_i0_valid = (v_i0 >= 0.f);
            auto v_i1_valid = (v_i1 < static_cast<float>(proj_height));

            auto upper_row = reinterpret_cast<const float*>(reinterpret_cast<const char*>(proj) + v_i0_ui * proj_pitch);
            auto lower_row = reinterpret_cast<const float*>(reinterpret_cast<const char*>(proj) + v_i1_ui * proj_pitch);

            // no bounds checking needed as tex2D automatically clamps to 0 outside of its bounds
            /*auto tl = tex2D<float>(proj, h_j0, v_i0);
            auto bl = tex2D<float>(proj, h_j0, v_i1);
            auto tr = tex2D<float>(proj, h_j1, v_i0);
            auto br = tex2D<float>(proj, h_j1, v_i1);*/

            auto tl = 0.f;
            auto bl = 0.f;
            auto tr = 0.f;
            auto br = 0.f;
            if(h_j0_valid && h_j1_valid && v_i0_valid && v_i1_valid)
            {
                tl = upper_row[h_j0_ui];
                bl = lower_row[h_j0_ui];
                tr = upper_row[h_j1_ui];
                br = lower_row[h_j1_ui];
            }

            auto val =  w_h1    * w_v1  * tl +
                        w_h1    * w_v0  * bl +
                        w_h0    * w_v1  * tr +
                        w_h0    * w_v0  * br;

            return val;
        }

        __global__ void backproject(float* __restrict__ vol, std::size_t vol_w, std::size_t vol_h, std::size_t vol_d, std::size_t vol_pitch,
                                    std::size_t vol_offset, std::size_t vol_d_full, float voxel_size_x, float voxel_size_y, float voxel_size_z,
                                    cudaTextureObject_t proj, std::size_t proj_w, std::size_t proj_h, std::size_t proj_pitch,
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

                // add offset for the current subvolume
                auto m_off = m + vol_offset;

                // get centered coordinates -- volume center is at (0, 0, 0) and the top slice is at -(vol_d_off / 2)
                auto x_k = vol_centered_coordinate(k, vol_w, voxel_size_x);
                auto y_l = vol_centered_coordinate(l, vol_h, voxel_size_y);
                auto z_m = vol_centered_coordinate(m_off, vol_d_full, voxel_size_z);

                // rotate coordinates
                auto s = x_k * angle_cos + y_l * angle_sin;
                auto t = -x_k * angle_sin + y_l * angle_cos;
                auto z = z_m;

                // project rotated coordinates
                auto factor = dist_sd / (s + dist_src);
                auto h = proj_real_coordinate(t * factor, proj_w, pixel_size_x, pixel_offset_x);
                auto v = proj_real_coordinate(z * factor, proj_h, pixel_size_y, pixel_offset_y);
                //auto h = t * factor;
                //auto v = z * factor;

                // get projection value by interpolation
                // auto det = interpolate(h, v, proj, proj_w, proj_h, proj_pitch, pixel_size_x, pixel_size_y, pixel_offset_x, pixel_offset_y);
                /*auto det = 0.f;
                if((h < proj_w) && (v < proj_h))
                    det = tex2D<float>(proj, h, v);*/

                auto det = tex2D<float>(proj, h, v);

                // backproject
                auto u = -(dist_src / (s + dist_src));
                row[k] += 0.5f * det * powf(u, 2.f);
            }
        }
    }

    reconstruction_stage::reconstruction_stage(const geometry& det_geo, const volume_type& vol_geo, const std::vector<volume_type>& subvol_geos,
                                                bool predefined_angles)
    : det_geo_(det_geo)
    , vol_geo_{nullptr, vol_geo.width, vol_geo.height, vol_geo.depth, vol_geo.remainder, vol_geo.offset, vol_geo.valid, vol_geo.device, vol_geo.vx_size_x, vol_geo.vx_size_y, vol_geo.vx_size_z}
    , predefined_angles_{predefined_angles}
    {
        auto sce = stage_construction_error{"reconstruction_stage::reconstruction_stage() failed"};

        try
        {
            devices_ = ddrf::cuda::get_device_count();

            auto d_sv = static_cast<svv_size_type>(devices_);
            subvol_vec_ = decltype(subvol_vec_){d_sv};

            auto d_svg = static_cast<svgv_size_type>(devices_);
            subvol_geo_vec_ = decltype(subvol_geo_vec_){d_svg};

            auto d_iv = static_cast<iv_size_type>(devices_);
            input_vec_ = decltype(input_vec_){d_iv};

            vol_out_ = {ddrf::cuda::make_unique_pinned_host<float>(vol_geo_.width, vol_geo_.height, vol_geo_.depth),
                        vol_geo_.width, vol_geo_.height, vol_geo_.depth, vol_geo_.remainder, vol_geo_.offset, true, 0,
                        vol_geo_.vx_size_x, vol_geo_.vx_size_y, vol_geo_.vx_size_z};
            ddrf::cuda::fill(ddrf::cuda::sync, vol_out_.ptr, 0, vol_out_.width, vol_out_.height, vol_out_.depth);

            for(auto i = 0; i < devices_; ++i)
            {
                ddrf::cuda::set_device(i);

                for(const auto& g : subvol_geos)
                {
                    if(g.device == i)
                    {
                        auto ptr = ddrf::cuda::make_unique_device<float>(g.width, g.height, g.depth + g.remainder);
                        ddrf::cuda::fill(ddrf::cuda::sync, ptr, 0, g.width, g.height, g.depth + g.remainder);

                        d_sv = static_cast<svv_size_type>(g.device);
                        subvol_vec_[d_sv] = volume_type{std::move(ptr), g.width, g.height, g.depth, g.remainder, g.offset, true, g.device,
                                                        g.vx_size_x, g.vx_size_y, g.vx_size_z};

                        d_svg = static_cast<svgv_size_type>(g.device);
                        subvol_geo_vec_[d_svg] = volume_type{nullptr, g.width, g.height, g.depth, g.remainder, g.offset, true, g.device,
                            g.vx_size_x, g.vx_size_y, g.vx_size_z};
                        break;
                    }
                }
            }
        }
        catch(const ddrf::cuda::bad_alloc& ba)
        {
            BOOST_LOG_TRIVIAL(fatal) << "reconstruction_stage::reconstruction_stage() encountered a bad_alloc: " << ba.what();
            throw sce;
        }
        catch(const ddrf::cuda::invalid_argument& ia)
        {
            BOOST_LOG_TRIVIAL(fatal) << "reconstruction_stage::reconstruction_stage() passed an invalid argument to the CUDA runtime: " << ia.what();
            throw sce;
        }
        catch(const ddrf::cuda::runtime_error& re)
        {
            BOOST_LOG_TRIVIAL(fatal) << "reconstruction_stage::reconstruction_stage() caused a CUDA runtime error: " << re.what();
            throw sce;
        }
    }

    reconstruction_stage::~reconstruction_stage()
    {
        for(auto&& s : subvol_vec_)
        {
            cudaSetDevice(s.device);
            s.ptr.reset(nullptr);
        }
    }

    auto reconstruction_stage::run() -> void
    {
        std::vector<std::future<void>> futures;
        for(int i = 0; i < devices_; ++i)
            futures.emplace_back(std::async(std::launch::async, &reconstruction_stage::process, this, i));

        while(true)
        {
            auto proj = input_();
            auto valid = proj.valid;
            safe_push(std::move(proj));
            if(!valid)
                break;
        }

        for(auto&& f: futures)
            f.get();

        output_(std::move(vol_out_));
        BOOST_LOG_TRIVIAL(info) << "Reconstruction complete.";
    }

    auto reconstruction_stage::set_input_function(std::function<input_type(void)> input) noexcept -> void
    {
        input_ = input;
    }

    auto reconstruction_stage::set_output_function(std::function<void(output_type)> output) noexcept -> void
    {
        output_ = output;
    }

    auto reconstruction_stage::safe_push(input_type proj) -> void
    {
        auto&& lock = std::lock_guard<std::mutex>{mutex__};

        if(proj.valid)
            input_vec_[proj.device].push(std::move(proj));
        else
        {
            for(auto i = 0; i < devices_; ++i)
                input_vec_[i].push(input_type{});
        }
    }

    auto reconstruction_stage::safe_pop(int device) -> input_type
    {
        while(input_vec_.empty())
            std::this_thread::yield();

        auto d = static_cast<iv_size_type>(device);
        auto&& queue = input_vec_.at(d);
        while(queue.empty())
            std::this_thread::yield();

        auto&& lock = std::lock_guard<std::mutex>{mutex__};

        auto proj = std::move(queue.front());
        queue.pop();

        return proj;
    }

    auto reconstruction_stage::process(int device) -> void
    {
        auto sre = stage_runtime_error{"reconstruction_stage::process() failed"};
        try
        {
            ddrf::cuda::set_device(device);

            auto vol_count = svgv_size_type{0};
            auto first = true;

            auto delta_s = det_geo_.delta_s * det_geo_.l_px_row;
            auto delta_t = det_geo_.delta_t * det_geo_.l_px_col;
            while(true)
            {
                auto d_svg = static_cast<svgv_size_type>(device);
                auto& v_geo = subvol_geo_vec_.at(d_svg);

                auto d_sv = static_cast<svv_size_type>(device);
                auto& v = subvol_vec_.at(d_sv);

                auto p = safe_pop(device);
                if(!p.valid)
                {
                    download_and_reset(device, vol_count);
                    break;
                }

                if(p.idx == 0)
                {
                    if(first)
                        first = false;
                    else
                    {
                        download_and_reset(device, vol_count);
                        ++vol_count;
                    }
                }

                if(p.idx % 10 == 0)
                    BOOST_LOG_TRIVIAL(info) << "Reconstruction processing projection #" << p.idx << " on device #" << device << " in stream " << p.stream;

                auto phi = 0.f;
                if(predefined_angles_)
                    phi = p.phi;
                else
                    phi = static_cast<float>(p.idx) * det_geo_.delta_phi;
                auto phi_rad = phi * M_PI / 180.f;
                auto sin = static_cast<float>(std::sin(phi_rad));
                auto cos = static_cast<float>(std::cos(phi_rad));

                auto offset = v_geo.offset * vol_count;

                auto v_ptr = v.ptr.get();
                auto p_ptr = p.ptr.get();
                auto p_cptr = static_cast<const float*>(p.ptr.get());

                // transform the projection into a texture
                auto res_desc = cudaResourceDesc{};
                res_desc.resType = cudaResourceTypePitch2D;
                res_desc.res.pitch2D.desc = cudaCreateChannelDesc<float>();
                res_desc.res.pitch2D.devPtr = reinterpret_cast<void*>(p_ptr);
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

                ddrf::cuda::launch_async(p.stream, v.width, v.height, v.depth,
                                    backproject,
                                    v_ptr, v.width, v.height, v.depth, v.ptr.pitch(), offset, vol_geo_.depth,
                                        v.vx_size_x, v.vx_size_y, v.vx_size_z,
                                    tex, p.width, p.height, p.ptr.pitch(), det_geo_.l_px_row, det_geo_.l_px_col,
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
        }
        catch(const ddrf::cuda::bad_alloc& ba)
        {
            BOOST_LOG_TRIVIAL(fatal) << "reconstruction_stage::process() encountered a bad_alloc: " << ba.what();
            throw sre;
        }
        catch(const ddrf::cuda::invalid_argument& ia)
        {
            BOOST_LOG_TRIVIAL(fatal) << "reconstruction_stage::process() passed an invalid argument to the CUDA runtime: " << ia.what();
            throw sre;
        }
        catch(const ddrf::cuda::runtime_error& re)
        {
            BOOST_LOG_TRIVIAL(fatal) << "reconstruction-stage::process() encountered a CUDA runtime error: " << re.what();
            throw sre;
        }
    }

    auto reconstruction_stage::download_and_reset(int device, std::uint32_t vol_count) -> void
    {
        auto d_sv = static_cast<svv_size_type>(device);
        auto&& v = subvol_vec_.at(d_sv);

        ddrf::cuda::copy(ddrf::cuda::sync, vol_out_.ptr, v.ptr, v.width, v.height, v.depth + v.remainder,
                            0, 0, vol_count * v.offset);
        BOOST_LOG_TRIVIAL(debug) << "Copy succeeded";

        ddrf::cuda::fill(ddrf::cuda::sync, v.ptr, 0, v.width, v.height, v.depth);
        BOOST_LOG_TRIVIAL(debug) << "Memset succeeded";
    }
}
