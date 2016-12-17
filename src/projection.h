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

#ifndef PARIS_PROJECTION_H_
#define PARIS_PROJECTION_H_

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <utility>

#ifndef __NVCC__
#include <cuda_runtime.h>
#endif

#include <glados/cuda/memory.h>

namespace paris
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

#endif /* PARIS_PROJECTION_H_ */
