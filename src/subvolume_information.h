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
 * Date: 28 October 2016
 * Authors: Jan Stephan <j.stephan@hzdr.de>
 */

#ifndef DDAFA_SUBVOLUME_INFORMATION_H_
#define DDAFA_SUBVOLUME_INFORMATION_H_

#include "geometry.h"

namespace ddafa
{
    struct subvolume_info
    {
        subvolume_geometry geo;
        int num;
    };
}

#endif /* DDAFA_SUBVOLUME_INFORMATION_H_ */
