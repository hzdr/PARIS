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

#include <ddrf/cuda/exception.h>
#include <ddrf/cuda/memory.h>
#include <ddrf/cuda/utility.h>

#include "exception.h"
#include "geometry.h"
#include "geometry_calculator.h"
#include "volume.h"

namespace ddafa
{
    geometry_calculator::geometry_calculator(const geometry& geo, bool enable_roi,
                                                std::uint32_t roi_x1, std::uint32_t roi_x2,
                                                std::uint32_t roi_y1, std::uint32_t roi_y2,
                                                std::uint32_t roi_z1, std::uint32_t roi_z2)
    : det_geo_(geo)
    , vol_geo_{}
    , d_sd_{std::abs(det_geo_.d_od) + std::abs(det_geo_.d_so)}
    , vol_count_{0u}
    {
        auto sce = stage_construction_error{"geometry_calculator::geometry_calculator() failed"};

        try
        {
            devices_ = ddrf::cuda::get_device_count();

            calculate_volume_width_height_vx();
            calculate_volume_depth_vx();
            calculate_volume_height_mm();
            calculate_memory_footprint();
            calculate_volume_partition();
            calculate_subvolume_offsets();
        }
        catch(const ddrf::cuda::bad_alloc& ba)
        {
            BOOST_LOG_TRIVIAL(fatal) << "geometry_calculator::geometry_calculator() encountered a bad_alloc: " << ba.what();
            throw sce;
        }
        catch(const ddrf::cuda::invalid_argument& ia)
        {
            BOOST_LOG_TRIVIAL(fatal) << "geometry_calculator::geometry_calculator() passed an invalid argument to the CUDA runtime: " << ia.what();
            throw sce;
        }
        catch(const ddrf::cuda::runtime_error& re)
        {
            BOOST_LOG_TRIVIAL(fatal) << "geometry_calculator::geometry_calculator() caused a CUDA runtime error: " << re.what();
            throw sce;
        }
    }

    auto geometry_calculator::get_projection_iteration_num() const noexcept -> std::uint32_t
    {
        using pair_type = typename decltype(vol_per_dev_)::value_type;

        auto p = std::max_element(std::begin(vol_per_dev_), std::end(vol_per_dev_), [](const pair_type& p1, const pair_type& p2) { return p1.second < p2.second; });
        return p->second;
    }

    auto geometry_calculator::get_volume_metadata() const noexcept -> volume_type
    {
        return volume_type{nullptr, vol_geo_.width, vol_geo_.height, vol_geo_.depth, vol_geo_.remainder, vol_geo_.offset, vol_geo_.valid, vol_geo_.device,
                            vol_geo_.vx_size_x, vol_geo_.vx_size_y, vol_geo_.vx_size_z};
    }

    auto geometry_calculator::get_subvolume_metadata() const noexcept -> std::vector<volume_type>
    {
        auto vec = std::vector<volume_type>{};
        for(const auto& p : vol_geo_per_dev_)
            vec.emplace_back(nullptr, p.second.width, p.second.height, p.second.depth, p.second.remainder, p.second.offset, p.second.valid, p.second.device,
                            p.second.vx_size_x, p.second.vx_size_y, p.second.vx_size_z);

        return vec;
    }

    auto geometry_calculator::calculate_memory_footprint() noexcept -> void
    {
        // this is not entirely accurate as CUDA adds a few more bytes when allocating 2D/3D memory
        vol_mem_ = vol_geo_.width * vol_geo_.height * vol_geo_.depth * sizeof(float);
        proj_mem_ = det_geo_.n_row * det_geo_.n_col * sizeof(float);

        BOOST_LOG_TRIVIAL(info) << "The volume requires (roughly) " << vol_mem_ << " bytes";
        BOOST_LOG_TRIVIAL(info) << "One projection requires (roughly) " << proj_mem_ << " bytes";
    }

    auto geometry_calculator::calculate_volume_partition() -> void
    {
        auto d = static_cast<std::size_t>(devices_);
        vol_mem_ /= d;

        for(auto i = 0; i < devices_; ++i)
        {
            auto vol_mem_dev = vol_mem_;
            auto required_mem = vol_mem_dev + 32 * proj_mem_;
            auto vol_count_dev = 1u;

            ddrf::cuda::set_device(i);

            auto free_mem = std::size_t{};
            auto total_mem = std::size_t{};
            ddrf::cuda::get_memory_info(free_mem, total_mem);

            auto calc_volume_size = std::function<std::size_t(std::size_t, std::size_t, std::uint32_t*, std::size_t)>{};
            calc_volume_size = [&calc_volume_size, this](std::size_t req_mem, std::size_t vol_mem, std::uint32_t* vol_count, std::size_t dev_mem)
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

            vol_geo_per_dev_[i] = volume_type{nullptr, vol_geo_.width, vol_geo_.height, (vol_geo_.depth / d) / vol_count_dev, 0, 0, false, i,
                                                vol_geo_.vx_size_x, vol_geo_.vx_size_y, vol_geo_.vx_size_z};
        }
    }

    auto geometry_calculator::calculate_subvolume_offsets() noexcept -> void
    {
        auto d = static_cast<std::size_t>(devices_);
        for(auto i = 0; i < devices_; ++i)
        {
            if(vol_per_dev_.count(i) == 0)
                continue;

            BOOST_LOG_TRIVIAL(debug) << "d: " << d;
            BOOST_LOG_TRIVIAL(debug) << "vol_per_dev_[" << i << "]: " << vol_per_dev_[i];
            vol_geo_per_dev_[i].offset = (vol_geo_.depth / d) / vol_per_dev_[i];

            auto r1 = vol_geo_.depth % d; // remainder when partitioning amongst the devices
            auto r2 = (vol_geo_.depth / d) % vol_per_dev_[i]; // remainder when partitioning on one device
            BOOST_LOG_TRIVIAL(debug) << "r1: " << r1;
            BOOST_LOG_TRIVIAL(debug) << "r2: " << r2;
            vol_geo_per_dev_[i].remainder = r2;
            if(i == (devices_ - 1))
                vol_geo_per_dev_[i].remainder += r1;

            BOOST_LOG_TRIVIAL(info) << "Volume offset on device #" << i << ": " << vol_geo_per_dev_[i].offset;
            BOOST_LOG_TRIVIAL(info) << "Remainder on device #" << i << ": " << vol_geo_per_dev_[i].remainder;
        }
    }
}
