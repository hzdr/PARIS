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
 * Date: 02 December 2016
 * Authors: Jan Stephan <j.stephan@hzdr.de>
 */

#include <glados/cuda/utility.h>

#include "backend.h"

namespace paris
{
    namespace cuda
    {
        auto make_async_handle() -> async_handle
        {
            return glados::cuda::create_concurrent_stream();
        }

        auto destroy_async_handle(async_handle& handle) noexcept -> error_type
        {
            return cudaStreamDestroy(handle);
        }

        auto synchronize(const async_handle& handle) -> void
        {
            glados::cuda::synchronize_stream(handle);
        }
    }
}
