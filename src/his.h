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
 * Date: 18 August 2016
 * Authors: Jan Stephan
 */

#ifndef DDAFA_HIS_LOADER_H_
#define DDAFA_HIS_LOADER_H_

#include <string>
#include <utility>
#include <vector>

#include "backend.h"
#include "projection.h"

namespace ddafa
{
    namespace his
    {
        using image_type = projection<backend::host_ptr_2D>;
        auto load(const std::string& path) -> std::vector<image_type>;
    }
}



#endif /* DDAFA_HIS_LOADER_H_ */
