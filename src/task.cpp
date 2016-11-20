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
 * Date: 10 November 2016
 * Authors: Jan Stephan
 */

#include <cstdint>
#include <queue>

#include "geometry.h"
#include "program_options.h"
#include "scheduler.h"
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
