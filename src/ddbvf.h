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
 * Date: 21 November 2016
 * Authors: Jan Stephan
 */

#ifndef DDAFA_DDBVF_H_
#define DDAFA_DDBVF_H_

#include <cstdint>
#include <memory>
#include <string>

#include <ddrf/cuda/memory.h>

#include "volume.h"

namespace ddafa
{
    namespace ddbvf
    {
        struct handle;
        struct handle_deleter { auto operator()(handle* h) noexcept -> void; };
        using handle_type = std::unique_ptr<handle, handle_deleter>;

        using volume_type = volume<ddrf::cuda::pinned_host_ptr<float>>;

        auto open(const std::string& path) -> handle_type;
        auto create(const std::string& path, std::uint32_t dim_x, std::uint32_t dim_y, std::uint32_t dim_z) -> handle_type;

        auto write(handle_type& h, const volume_type& vol, std::uint32_t first) -> void;
        auto read(handle_type& h, std::uint32_t first) -> volume_type;
        auto read(handle_type& h, std::uint32_t first, std::uint32_t last) -> volume_type;

    }
}

#endif /* DDAFA_DDBVF_H_ */
