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
 * Date: 09 September 2016
 * Authors: Jan Stephan
 */

#ifndef DDAFA_PROJECTION_H_
#define DDAFA_PROJECTION_H_

#include <cstddef>
#include <cstdint>
#include <utility>

namespace ddafa
{
    template <class Ptr>
    struct projection
    {
        projection() noexcept = default;

        projection(Ptr p, std::size_t w, std::size_t h, std::uint32_t i, float ph, bool v, int dev) noexcept
        : ptr{std::move(p)}, width{w}, height{h}, idx{i}, phi{ph}, valid{v}, device{dev}
        {}

        Ptr ptr = nullptr;
        std::size_t width = 0;
        std::size_t height = 0;
        std::uint32_t idx = 0;
        float phi = 0.f;
        bool valid = false;
        int device = 0;
    };
}

#endif /* DDAFA_PROJECTION_H_ */
