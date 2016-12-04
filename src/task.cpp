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
 * Date: 10 November 2016
 * Authors: Jan Stephan <j.stephan@hzdr.de>
 */

#include <cstdint>
#include <queue>

#include "geometry.h"
#include "program_options.h"
#include "subvolume_information.h"
#include "task.h"

namespace ddafa
{
    auto make_tasks(const program_options& po, const volume_geometry& vol_geo, const subvolume_info& subvol_info)
    -> std::queue<task>
    {
        auto q = std::queue<task>{};

        for(auto i = 0; i < subvol_info.num; ++i)
        {
            auto subvol_geo = subvol_info.geo;
            q.emplace(task{static_cast<std::uint32_t>(i),
                            static_cast<std::uint32_t>(subvol_info.num),
                            po.input_path,
                            po.det_geo, vol_geo, subvol_geo,
                            po.enable_roi, po.roi,
                            po.enable_angles, po.angle_path,
                            po.quality});
        }

        return q;
    }
}
