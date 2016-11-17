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
    };

    auto make_program_options(int argc, char** argv) -> program_options;
}



#endif /* DDAFA_PROGRAM_OPTIONS_H_ */