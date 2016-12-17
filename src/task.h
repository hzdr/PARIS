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
 * Date: 07 November 2016
 * Authors: Jan Stephan <j.stephan@hzdr.de>
 */

#ifndef PARIS_TASK_H_
#define PARIS_TASK_H_

#include <cstdint>
#include <string>
#include <queue>

#include "geometry.h"
#include "program_options.h"
#include "region_of_interest.h"
#include "subvolume_information.h"

namespace paris
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



#endif /* PARIS_TASK_H_ */
