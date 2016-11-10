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

#ifndef DDAFA_REGION_OF_INTEREST_H_
#define DDAFA_REGION_OF_INTEREST_H_

#include <cstdint>

namespace ddafa
{
    struct region_of_interest
    {
        std::uint32_t x1;
        std::uint32_t x2;
        std::uint32_t y1;
        std::uint32_t y2;
        std::uint32_t z1;
        std::uint32_t z2;
    };
}



#endif /* REGION_OF_INTEREST_H_ */
