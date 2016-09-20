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
 * Date: 24 August 2016
 * Authors: Jan Stephan
 */

#include <functional>
#include <utility>

#include <ddrf/cuda/memory.h>
#include <ddrf/memory.h>

#include "projection.h"

namespace ddafa
{
    class device_to_host_stage
    {
        private:
            using device_allocator = ddrf::cuda::device_allocator<float, ddrf::memory_layout::pointer_2D>;
            using pool_allocator = ddrf::pool_allocator<float, ddrf::memory_layout::pointer_2D, device_allocator>;
            using smart_pointer = typename pool_allocator::smart_pointer;

        public:
            using input_type = projection<smart_pointer>;
            using output_type = projection<ddrf::cuda::pinned_host_ptr<float>>;

        public:
            auto run() -> void;
            auto set_input_function(std::function<input_type(void)> input) noexcept -> void;
            auto set_output_function(std::function<void(output_type)> output) noexcept -> void;

        private:
            std::function<input_type(void)> input_;
            std::function<void(output_type)> output_;
    };
}
