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
 * Date: 18 August 2016
 * Authors: Jan Stephan
 */

#ifndef DDAFA_WEIGHTING_STAGE_H_
#define DDAFA_WEIGHTING_STAGE_H_

#include <cstdint>
#include <functional>

#include <ddrf/cuda/memory.h>
#include <ddrf/memory.h>

#include "geometry.h"
#include "projection.h"
#include "task.h"

namespace ddafa
{
    class weighting_stage
    {
        private:
            using device_allocator = ddrf::cuda::device_allocator<float, ddrf::memory_layout::pointer_2D>;
            using pool_allocator = ddrf::pool_allocator<float, ddrf::memory_layout::pointer_2D, device_allocator>;
            using smart_pointer = typename pool_allocator::smart_pointer;

        public:
            using input_type = projection<smart_pointer>;
            using output_type = projection<smart_pointer>;

        public:
            weighting_stage(int device) noexcept;

            auto assign_task(task t) noexcept -> void;
            auto run() const -> void;
            auto set_input_function(std::function<input_type(void)> input) noexcept -> void;
            auto set_output_function(std::function<void(output_type)> output) noexcept -> void;

        private:
            std::function<input_type(void)> input_;
            std::function<void(output_type)> output_;

            int device_;

            detector_geometry det_geo_;
            float h_min_;
            float v_min_;
            float d_sd_;
    };
}

#endif /* DDAFA_WEIGHTING_STAGE_H_ */
