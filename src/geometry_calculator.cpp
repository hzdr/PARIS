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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iterator>
#include <string>
#include <vector>

#include <boost/log/trivial.hpp>

#include <cuda_runtime.h>

#include <ddrf/cuda/memory.h>

#include "exception.h"
#include "geometry.h"
#include "geometry_calculator.h"
#include "metadata.h"

namespace ddafa
{
    geometry_calculator::geometry_calculator(const geometry& geo)
    : det_geo_{geo}
    , vol_geo_{0}
    , d_sd_{std::abs(det_geo_.d_od) + std::abs(det_geo_.d_so)}
    , vol_count_{0u}
    {
        auto err = cudaGetDeviceCount(&devices_);
        if(err != cudaSuccess)
        {
            BOOST_LOG_TRIVIAL(fatal) << "geometry_calculator::geometry_calculator() could not obtain CUDA devices: " << cudaGetErrorString(err);
            throw stage_construction_error{"geometry_calculator::geometry_calculator() failed"};
        }

        calculate_volume_width_height_vx();
        calculate_volume_depth_vx();
        calculate_volume_height_mm();
        calculate_memory_footprint();
        calculate_volume_partition();
        calculate_subvolume_offsets();
    }

    auto geometry_calculator::get_projection_iteration_num() const noexcept -> std::uint32_t
    {
        using pair_type = typename decltype(vol_per_dev_)::value_type;

        auto p = std::max_element(std::begin(vol_per_dev_), std::end(vol_per_dev_), [](const pair_type& p1, const pair_type& p2) { return p1.second < p2.second; });
        return p->second;
    }

    auto geometry_calculator::get_volume_metadata() const noexcept -> volume_metadata
    {
        return vol_geo_;
    }

    auto geometry_calculator::get_subvolume_metadata() const noexcept -> std::vector<volume_metadata>
    {
        auto vec = std::vector<volume_metadata>{};
        for(const auto& p : vol_geo_per_dev_)
            vec.push_back(p.second);

        return vec;
    }

    auto geometry_calculator::calculate_volume_width_height_vx() noexcept -> void
    {
        auto n_row = float{det_geo_.n_row};
        auto l_px_row = det_geo_.l_px_row;
        auto delta_s = det_geo_.delta_s * l_px_row; // the offset is originally measured in pixels!

        auto alpha = std::atan((((n_row * l_px_row) / 2.f) + std::abs(delta_s)) / d_sd_);
        auto r = std::abs(det_geo_.d_so) * std::sin(alpha);

        vol_geo_.vx_size_x = r / ((((n_row * l_px_row) / 2.f) + std::abs(delta_s)) / l_px_row);
        vol_geo_.vx_size_y = vol_geo_.vx_size_x;

        vol_geo_.width = std::size_t{(2.f * r) / vol_geo_.vx_size_x};
        vol_geo_.height = vol_geo_.width;
    }

    auto geometry_calculator::calculate_volume_depth_vx() noexcept -> void
    {
        vol_geo_.vx_size_z = vol_geo_.vx_size_x;
        auto n_col = float{det_geo_.n_col};
        auto l_px_col = det_geo_.l_px_col;
        auto delta_t = det_geo_.delta_t * l_px_col;

        vol_geo_.depth = std::size_t{((n_col * l_px_col / 2.f) + std::abs(delta_t)) * (std::abs(det_geo_.d_so) / d_sd_) * (2.f / vol_geo_.vx_size_z)};

        BOOST_LOG_TRIVIAL(info) << "Volume dimensions: " << vol_geo_.width << " x " << vol_geo_.height << " x " << vol_geo_.depth << " vx";
        BOOST_LOG_TRIVIAL(info) << "Voxel size: " << std::setprecision(4) << vol_geo_.vx_size_x << " x " << vol_geo_.vx_size_y << " x " << vol_geo_.vx_size_z << " mm";
    }

    auto geometry_calculator::calculate_volume_height_mm() noexcept -> void
    {
        vol_height_ = vol_geo_.depth * vol_geo_.vx_size_z;
        BOOST_LOG_TRIVIAL(info) << "Volume height: " << volume_height_ << " mm";
    }

    auto geometry_calculator::calculate_memory_footprint() noexcept -> void
    {
        // this is not entirely accurate as CUDA adds a few more bytes when allocating 2D/3D memory
        vol_mem_ = vol_geo_.width * vol_geo_.height * vol_geo_.depth * sizeof(float);
        proj_mem_ = det_geo_.n_row * det_geo_.n_col * sizeof(float);

        BOOST_LOG_TRIVIAL(info) << "The volume requires (roughly) " << vol_mem_ << " bytes";
        BOOST_LOG_TRIVIAL(info) << "One projection requires (roughly)" << proj_mem_ << " bytes";
    }

    auto geometry_calculator::calculate_volume_partition() -> void
    {
        vol_mem_ /= std::size_t{devices_};
        for(auto i = 0; i < devices_; ++i)
        {
            auto vol_mem_dev = vol_mem_;
            auto required_mem = vol_mem_dev + 32 * proj_mem_;
            auto vol_count_dev = 1u;
            auto err = cudaSetDevice(i);
            if(err != cudaSuccess)
            {
                BOOST_LOG_TRIVIAL(fatal) << "geometry_calculator::calculate_volume_partition() could not set CUDA device: " << cudaGetErrorString(err);
                throw stage_construction_error{"geometry_calculator::calculate_volume_partition() failed"};
            }

            auto free_mem = std::size_t{};
            auto total_mem = std::size_t{};
            err = cudaMemGetInfo(&free_mem, &total_mem);
            if(err != cudaSuccess)
            {
                BOOST_LOG_TRIVIAL(fatal) << "geometry_calculator::calculate_volume_partition() could not get CUDA memory info: " << cudaGetErrorString(err);
                throw stage_construction_error{"geometry_calculator::calculate_volume_partition() failed"};
            }

            auto calc_volume_size = std::function<std::size_t(std::size_t, std::size_t, std::uint32_t*, std::size_t)>;
            calc_volume_size = [&calc_volume_size, this](std::size_t req_mem, std::size_t vol_mem, std::uint32* vol_count, std::size_t dev_mem)
            {
                if(req_mem >= dev_mem)
                {
                    vol_mem /= 2;
                    *vol_count *= 2;
                    req_mem = vol_mem + 32 * this->proj_mem_;
                    return calc_volume_size(req_mem, vol_mem, vol_count, dev_mem);
                }
                else
                    return req_mem;
            };

            required_mem = calc_volume_size(required_mem, vol_mem_dev, &vol_count_dev, free_mem);
            BOOST_LOG_TRIVIAL(info) << "Trying test allocation...";
            try
            {
                auto d = std::size_t{devices_};
                auto tmp = ddrf::cuda::make_unique_device<float>(vol_geo_.width, vol_geo_.height, (vol_geo_.depth / d) / vol_count_dev);
                BOOST_LOG_TRIVIAL(info) << "Test allocation successful.";
            }
            catch(const ddrf::cuda::bad_alloc& ba)
            {
                BOOST_LOG_TRIVIAL(info) << "Test allocation failed, reducing subvolume size.";
                vol_mem_dev /= 2;
                vol_count_dev *= 2;
                required_mem = vol_mem_dev + 32 * proj_mem_;
            }

            vol_count_ += vol_count_dev;
            auto chunk_str = std::string{vol_count_dev > 1 ? "chunks" : "chunk"};
            BOOST_LOG_TRIVIAL(info) << "The reconstruction requires " << vol_count_dev << " " << chunk_str << " with " << required_mem << " bytes on device " << i;
            vol_per_dev_[i] = vol_count_dev;
            auto vm = volume_metadata{vol_geo_.width, vol_geo_.height, (vol_geo_.depth / d) / vol_count_dev, 0, 0, false, i, vol_geo_.vx_size_x, vol_geo_.vx_size_y, vol_geo_.vx_size_z};
            vol_geo_per_dev_[i] = vm;
        }
    }

    auto geometry_calculator::calculate_subvolume_offsets() noexcept -> void
    {
        auto d = std::size_t{devices_};
        for(auto i = 0; i < devices_; ++i)
        {

            if(vol_per_dev_.count(i) == 0)
                continue;

            vol_geo_per_dev_[i].offset = (vol_geo_.depth / d) / vol_per_dev_[i];

            auto r1 = vol_geo_.depth % d; // remainder when partitioning amongst the devices
            auto r2 = (vol_geo_.depth / d) % vol_per_dev_[i]; // remainder when partitioning on one device
            vol_geo_per_dev_[i].remainder = r2;
            if(i == (devices_ - 1))
                vol_geo_per_dev_[i].remainder += r1;
        }
    }
}
