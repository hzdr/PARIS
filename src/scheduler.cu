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
 * Date: 28 October 2016
 * Authors: Jan Stephan
 */

#include <algorithm>
#include <map>

#include <boost/log/trivial.hpp>

#include <ddrf/cuda/exception.h>
#include <ddrf/cuda/memory.h>
#include <ddrf/cuda/utility.h>

#include "exception.h"
#include "geometry.h"
#include "scheduler.h"

namespace ddafa
{
    namespace
    {
        struct mem_info
        {
            std::size_t vol;
            std::size_t proj;
        };

        auto memory_info(const volume_geometry& vol_geo, const detector_geometry& det_geo) noexcept -> mem_info
        {
            auto info = mem_info{};
            // this is not entirely accurate as CUDA adds a few more bytes when allocating 2D/3D memory
            info.vol = vol_geo.dim_x * vol_geo.dim_y * vol_geo.dim_z * sizeof(float);
            info.proj = det_geo.n_row * det_geo.n_col * sizeof(float);

            BOOST_LOG_TRIVIAL(info) << "The volume requires (roughly) " << info.vol << " bytes";
            BOOST_LOG_TRIVIAL(info) << "One projection requires (roughly) " << info.proj << " bytes";

            return info;
        }
    }

    auto create_subvolume_information(const volume_geometry& vol_geo, const detector_geometry& det_geo, int proj_num) -> subvolume_info
    {
        auto sce = ddafa::stage_construction_error{"create_subvolume_information() failed"};

        try
        {
            auto subvol_info = subvolume_info{};
            auto info = memory_info(vol_geo, det_geo);
            auto mem_needed = info.vol + proj_num * info.proj;

            auto devices = ddrf::cuda::get_device_count();
            mem_needed /= devices;

            auto vols_needed = devices;

            for(auto d = 0; d < devices; ++d)
            {
                ddrf::cuda::set_device(d);

                auto mem_dev = mem_needed;

                auto mem_free = std::size_t{};
                auto mem_total = std::size_t{};
                ddrf::cuda::get_memory_info(mem_free, mem_total);

                while(std::max(mem_dev, mem_free) == mem_dev)
                {
                    mem_dev /= 2;
                    vols_needed *= 2;
                }

                // FIXME: nasty exception abuse
                try
                {
                    ddrf::cuda::make_unique_device<float>(vol_geo.dim_x, vol_geo.dim_y, vol_geo.dim_z / vols_needed);
                    BOOST_LOG_TRIVIAL(info) << "Test allocation successful.";
                }
                catch(const ddrf::cuda::bad_alloc& ba)
                {
                    BOOST_LOG_TRIVIAL(info) << "Test allocation failed, reducing subvolume size.";
                    mem_dev /= 2;
                    vols_needed *= 2;
                }
            }

            subvol_info.geo.dim_x = vol_geo.dim_x;
            subvol_info.geo.dim_y = vol_geo.dim_y;
            subvol_info.geo.dim_z = vol_geo.dim_z / vols_needed;
            subvol_info.geo.remainder = vol_geo.dim_z % vols_needed;
            subvol_info.num = vols_needed;

            return subvol_info;
        }
        catch(const ddrf::cuda::bad_alloc& ba)
        {
            BOOST_LOG_TRIVIAL(fatal) << "create_subvolume_geometry() encountered a bad_alloc: " << ba.what();
            throw sce;
        }
        catch(const ddrf::cuda::invalid_argument& ia)
        {
            BOOST_LOG_TRIVIAL(fatal) << "create_subvolume_geometry() passed an invalid argument to the CUDA runtime: " << ia.what();
            throw sce;
        }
        catch(const ddrf::cuda::runtime_error& re)
        {
            BOOST_LOG_TRIVIAL(fatal) << "create_subvolume_geometry() caused a CUDA runtime error: " << re.what();
            throw sce;
        }
    }
}


