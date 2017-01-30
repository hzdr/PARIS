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
 * Date: 18 August 2016
 * Authors: Jan Stephan
 */

#ifndef PARIS_HIS_LOADER_H_
#define PARIS_HIS_LOADER_H_

#include <string>
#include <utility>
#include <vector>

#include "backend.h"
#include "projection.h"

namespace paris
{
    namespace his
    {
        using image_type = backend::projection_host_type;
        auto load(const std::string& path) -> std::vector<image_type>;
    }
}

#endif /* PARIS_HIS_LOADER_H_ */
