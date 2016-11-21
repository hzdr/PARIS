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
 * Date: 21 November 2016
 * Authors: Jan Stephan
 */

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
