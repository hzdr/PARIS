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

#ifndef PARIS_PRELOADER_STAGE_H_
#define PARIS_PRELOADER_STAGE_H_

#include <cstddef>
#include <functional>

#include <glados/memory.h>

#include "backend.h"
#include "projection.h"
#include "task.h"

namespace paris
{
    class preloader_stage
    {
        private:
            using pool_allocator = glados::pool_allocator<float, glados::memory_layout::pointer_2D, backend::allocator>;
            using smart_pointer = typename pool_allocator::smart_pointer;

        public:
            using input_type = projection<backend::host_ptr_2D<float>>;
            using output_type = projection<smart_pointer>;

        public:
            preloader_stage(std::size_t pool_limit, const backend::device_handle& device) noexcept;
            ~preloader_stage();
            preloader_stage(const preloader_stage& other);
            auto operator=(const preloader_stage& other) -> preloader_stage&;

            auto assign_task(task t) noexcept -> void;
            auto run() -> void;
            auto set_input_function(std::function<input_type(void)> input) noexcept -> void;
            auto set_output_function(std::function<void(output_type)> output) noexcept -> void;

        private:
            std::function<input_type(void)> input_;
            std::function<void(output_type)> output_;
            backend::device_handle device_;
            std::size_t limit_;
            pool_allocator pool_;
    };
}

#endif /* PARIS_PRELOADER_STAGE_H_ */
