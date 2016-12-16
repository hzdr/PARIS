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

#include <vector>

#include <glados/cuda/utility.h>

#include "backend.h"

namespace ddafa
{
    namespace cuda
    {
        auto set_device(const device_handle& device) -> void
        {
            glados::cuda::set_device(device);
        }

        auto set_device_noexcept(const device_handle& device) noexcept -> error_type
        {
            return cudaSetDevice(device);
        }

        auto get_devices() -> std::vector<device_handle>
        {
            auto vec = std::vector<device_handle>{};
            auto num = glados::cuda::get_device_count();

            for(auto i = 0; i < num; ++i)
                vec.push_back(i);

            return vec;
        }
    }
}
