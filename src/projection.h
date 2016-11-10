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
#include <cstdlib>
#include <utility>

#ifndef __NVCC__
#include <cuda_runtime.h>
#endif

namespace ddafa
{
    template <class Ptr>
    struct projection
    {
        projection() noexcept = default;

        projection(Ptr p, std::uint32_t w, std::uint32_t h, std::uint32_t i, float ph, bool v, cudaStream_t str) noexcept
        : ptr{std::move(p)}, width{w}, height{h}, idx{i}, phi{ph}, valid{v}, stream{str}
        {}

        projection(const projection&) = delete;
        auto operator=(const projection&) -> projection& = delete;

        projection(projection&& other) noexcept
        : ptr{std::move(other.ptr)}, width{other.width}, height{other.height}, idx{other.idx}, phi{other.phi}, valid{other.valid}
        , stream{other.stream}
        {
            other.ptr = nullptr;
            other.width = 0;
            other.height = 0;
            other.idx = 0;
            other.phi = 0.f;
            other.valid = false;
            other.stream = 0;
        }

        auto operator=(projection&& other) noexcept -> projection&
        {
            ptr = std::move(other.ptr);
            other.ptr = nullptr;

            width = other.width;
            other.width = 0;

            height = other.height;
            other.height = 0;

            idx = other.idx;
            other.idx = 0;

            phi = other.phi;
            other.phi = 0;

            valid = other.valid;
            other.valid = false;

            stream = other.stream;
            other.stream = 0;

            return *this;
        }

        ~projection()
        {
            if(stream != 0)
            {
                auto err = cudaStreamDestroy(stream);
                if(err != cudaSuccess)
                    std::exit(err);
            }
        }

        Ptr ptr = nullptr;
        std::uint32_t width = 0;
        std::uint32_t height = 0;
        std::uint32_t idx = 0;
        float phi = 0.f;
        bool valid = false;
        cudaStream_t stream = 0;
    };
}

#endif /* DDAFA_PROJECTION_H_ */
