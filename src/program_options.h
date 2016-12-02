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
 * Date: 07 November 2016
 * Authors: Jan Stephan <j.stephan@hzdr.de>
 */

#ifndef DDAFA_PROGRAM_OPTIONS_H_
#define DDAFA_PROGRAM_OPTIONS_H_

#include <cstdint>
#include <string>

#include "geometry.h"
#include "region_of_interest.h"

namespace ddafa
{
    struct program_options
    {
        detector_geometry det_geo;

        bool enable_io;
        std::string input_path;
        std::string output_path;
        std::string prefix;

        bool enable_roi;
        region_of_interest roi;

        bool enable_angles;
        std::string angle_path;

        std::uint16_t quality;
    };

    auto make_program_options(int argc, char** argv) -> program_options;
}



#endif /* DDAFA_PROGRAM_OPTIONS_H_ */
