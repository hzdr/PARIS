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

#ifndef PARIS_WEIGHTING_STAGE_H_
#define PARIS_WEIGHTING_STAGE_H_

#include <cstdint>
#include <functional>

#include <glados/memory.h>

#include "backend.h"
#include "geometry.h"
#include "projection.h"
#include "task.h"

namespace paris
{
    class weighting_stage
    {
        private:
            using pool_allocator = glados::pool_allocator<float, glados::memory_layout::pointer_2D, backend::allocator>;
            using smart_pointer = typename pool_allocator::smart_pointer;

        public:
            using input_type = projection<smart_pointer>;
            using output_type = projection<smart_pointer>;

        public:
            weighting_stage(const backend::device_handle& device) noexcept;

            auto assign_task(task t) noexcept -> void;
            auto run() const -> void;
            auto set_input_function(std::function<input_type(void)> input) noexcept -> void;
            auto set_output_function(std::function<void(output_type)> output) noexcept -> void;

        private:
            std::function<input_type(void)> input_;
            std::function<void(output_type)> output_;

            backend::device_handle device_;

            detector_geometry det_geo_;
            float h_min_;
            float v_min_;
            float d_sd_;
    };
}

#endif /* PARIS_WEIGHTING_STAGE_H_ */
