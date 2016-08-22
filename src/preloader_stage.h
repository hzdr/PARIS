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
 * Date: 18 August 2016
 * Authors: Jan Stephan
 */

#ifndef DDAFA_PRELOADER_STAGE_H_
#define DDAFA_PRELOADER_STAGE_H_

#include <functional>
#include <utility>

#include <ddrf/cuda/memory.h>
#include <ddrf/memory.h>

#include "metadata.h"

namespace ddafa
{
    class preloader_stage
    {
        private:
            using device_allocator = ddrf::cuda::device_allocator<float, ddrf::memory_layout::pointer_2D>;
            using host_allocator = ddrf::cuda::host_allocator<float, ddrf::memory_layout::pointer_2D>;
            using pool_allocator = ddrf::pool_allocator<float, ddrf::memory_layout::pointer_2D, device_allocator>;
            using host_pool_allocator = ddrf::pool_allocator<float, ddrf::memory_layout::pointer_2D, host_allocator>;
            using smart_pointer = typename pool_allocator::smart_pointer;
            using host_smart_pointer = typename host_pool_allocator::smart_pointer;

        public:
            using input_type = std::pair<host_smart_pointer, projection_metadata>;
            using output_type = std::pair<smart_pointer, projection_metadata>;

        public:
            preloader_stage() = default;
            auto run() -> void;
            auto set_input_function(std::function<input_type(void)> input) noexcept -> void;
            auto set_output_function(std::function<void(output_type)> output) noexcept -> void;

        private:
            std::function<input_type(void)> input_;
            std::function<void(output_type)> output_;
            pool_allocator alloc_;
    };
}

#endif /* DDAFA_PRELOADER_STAGE_H_ */
