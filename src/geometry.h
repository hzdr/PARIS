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

#ifndef DDAFA_GEOMETRY_H_
#define DDAFA_GEOMETRY_H_

#include <cstdint>

namespace ddafa
{
    struct geometry
    {
        // Detector
        std::uint32_t n_row;    // pixels per row
        std::uint32_t n_col;    // pixels per column
        float l_px_row;         // pixel size in horizontal direction [mm]
        float l_px_col;         // pixel size in vertical direction [mm]
        float delta_s;          // offset in horizontal direction [px]
        float delta_t;          // offset in vertical direction [px]

        // Support
        float d_so;             // distance source->object
        float d_od;             // distance object->detector

        // Rotation
        float delta_phi;        // difference between two successive angles - ignored if there is an angle file
    };
}

#endif /* DDAFA_GEOMETRY_H_ */
