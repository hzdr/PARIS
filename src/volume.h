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
 * Date: 09 September 2016
 * Authors: Jan Stephan <j.stephan@hzdr.de>
 */

#ifndef PARIS_VOLUME_H_
#define PARIS_VOLUME_H_

#include <cstdint>
#include <utility>

namespace paris
{
    template <typename BufferType, typename Metadata>
    struct volume
    {
        volume() = default;

        volume(BufferType b, std::uint32_t x, std::uint32_t y, std::uint32_t z, std::uint32_t o, Metadata m) 
        : buf(std::move(b)), dim_x{x}, dim_y{y}, dim_z{z}, off{o}, meta(m)
        {}

        BufferType buf = BufferType();
        std::uint32_t dim_x = 0;
        std::uint32_t dim_y = 0;
        std::uint32_t dim_z = 0;
        std::uint32_t off = 0;
        Metadata meta = Metadata();
    };
}

#endif /* PARIS_VOLUME_H_ */
