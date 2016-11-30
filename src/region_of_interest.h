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
