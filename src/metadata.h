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

#ifndef DDAFA_METADATA_H_
#define DDAFA_METADATA_H_

#include <cstddef>
#include <cstdint>

namespace ddafa
{
    struct projection_metadata
    {
        std::size_t width;
        std::size_t height;
        std::uint64_t index;
        float phi;
        bool valid = false;
        int device;
    };

    struct volume_metadata
    {
        std::size_t width;
        std::size_t height;
        std::size_t depth;
        std::size_t remainder;
        std::size_t offset;
        bool valid = false;
        int device;

        float vx_size_x;
        float vx_size_y;
        float vx_size_z;
    };
}

#endif /* DDAFA_METADATA_H_ */
