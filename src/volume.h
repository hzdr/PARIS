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

#include <cstddef>
#include <cstdint>
#include <utility>

namespace paris
{
    template <class Ptr>
    struct volume
    {
        volume() noexcept = default;

        volume(Ptr p, std::uint32_t w, std::uint32_t h, std::uint32_t d, std::uint32_t o, bool v, int dev) noexcept
        : ptr{std::move(p)}, width{w}, height{h}, depth{d}, offset{o}, valid{v}, device{dev}
        {}

        Ptr ptr = nullptr;
        std::uint32_t width = 0;
        std::uint32_t height = 0;
        std::uint32_t depth = 0;
        std::uint32_t offset = 0;
        bool valid = false;
        int device = 0;
    };
}

#endif /* PARIS_VOLUME_H_ */
