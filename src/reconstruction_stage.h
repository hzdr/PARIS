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
 * Date: 18 August 2016
 * Authors: Jan Stephan <j.stephan@hzdr.de>
 */

#ifndef PARIS_RECONSTRUCTION_STAGE_H_
#define PARIS_RECONSTRUCTION_STAGE_H_

#include <cstdint>
#include <functional>
#include <queue>
#include <utility>
#include <vector>

#include <glados/memory.h>

#include "backend.h"
#include "geometry.h"
#include "projection.h"
#include "region_of_interest.h"
#include "task.h"
#include "volume.h"

namespace paris
{
    class reconstruction_stage
    {
        private:
            using pool_allocator = glados::pool_allocator<float, glados::memory_layout::pointer_2D, backend::allocator>;
            using smart_pointer = typename pool_allocator::smart_pointer;

        public:
            using input_type = projection<smart_pointer>;
            using output_type = volume<backend::host_ptr_3D<float>>;
            using volume_type = volume<backend::device_ptr_3D<float>>;

        public:
            reconstruction_stage(const backend::device_handle& device) noexcept;
            ~reconstruction_stage() = default;

            auto assign_task(task t) noexcept -> void;
            auto run() -> void;
            auto set_input_function(std::function<input_type(void)> input) noexcept -> void;
            auto set_output_function(std::function<void(output_type)> output) noexcept -> void;

        private:
            auto process(int) -> void;
            auto download_and_reset(int, std::uint32_t) -> void;

        private:
            std::function<input_type(void)> input_;
            std::function<void(output_type)> output_;

            detector_geometry det_geo_;
            volume_geometry vol_geo_;
            subvolume_geometry subvol_geo_;

            bool enable_angles_;

            bool enable_roi_;
            region_of_interest roi_;
            volume_geometry roi_geo_;

            backend::device_handle device_;

            std::uint32_t task_id_;
            std::uint32_t task_num_;
    };
}

#endif /* PARIS_RECONSTRUCTION_STAGE_H_ */
