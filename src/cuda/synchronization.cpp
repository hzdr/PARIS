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
 * Date: 02 December 2016
 * Authors: Jan Stephan <j.stephan@hzdr.de>
 */

#include <ddrf/cuda/utility.h>

#include "backend.h"

namespace ddafa
{
    namespace cuda
    {
        auto make_async_handle() -> async_handle
        {
            return ddrf::cuda::create_concurrent_stream();
        }

        auto destroy_async_handle(async_handle& handle) noexcept -> error_type
        {
            return cudaStreamDestroy(handle);
        }

        auto synchronize(const async_handle& handle) -> void
        {
            ddrf::cuda::synchronize_stream(handle);
        }
    }
}
