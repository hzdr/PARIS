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

#ifndef PARIS_REGION_OF_INTEREST_H_
#define PARIS_REGION_OF_INTEREST_H_

#include <cstdint>

namespace paris
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



#endif /* PARIS_REGION_OF_INTEREST_H_ */
