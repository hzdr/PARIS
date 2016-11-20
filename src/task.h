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
 * Date: 07 November 2016
 * Authors: Jan Stephan
 */

#ifndef DDAFA_TASK_H_
#define DDAFA_TASK_H_

#include <cstdint>
#include <string>
#include <queue>

#include "geometry.h"
#include "program_options.h"
#include "region_of_interest.h"
#include "scheduler.h"

namespace ddafa
{
    struct task
    {
        std::uint32_t id;
        std::uint32_t num;

        std::string input_path;

        detector_geometry det_geo;
        volume_geometry vol_geo;
        subvolume_geometry subvol_geo;

        bool enable_roi;
        region_of_interest roi;

        bool enable_angles;
        std::string angle_path;
        
        std::uint16_t quality;
    };

    auto make_tasks(const program_options& po, const volume_geometry& vol_geo, const subvolume_info& subvol_info)
    -> std::queue<task>;
}



#endif /* DDAFA_TASK_H_ */
