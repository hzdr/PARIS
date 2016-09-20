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

#ifndef DDAFA_FILTER_STAGE_H_
#define DDAFA_FILTER_STAGE_H_

#include <cstdint>
#include <functional>
#include <map>
#include <queue>
#include <vector>

#include <ddrf/cuda/memory.h>
#include <ddrf/memory.h>

#include "projection.h"

namespace ddafa
{
    class filter_stage
    {
        private:
            using device_allocator = ddrf::cuda::device_allocator<float, ddrf::memory_layout::pointer_2D>;
            using pool_allocator = ddrf::pool_allocator<float, ddrf::memory_layout::pointer_2D, device_allocator>;
            using smart_pointer = typename pool_allocator::smart_pointer;

        public:
            using input_type = projection<smart_pointer>;
            using output_type = projection<smart_pointer>;

        public:
            filter_stage(std::uint32_t n_row, std::uint32_t n_col, float l_px_row);
            filter_stage(filter_stage&& other) = default;
            auto operator=(filter_stage&& other) -> filter_stage& = default;

            auto run() -> void;
            auto set_input_function(std::function<input_type(void)> input) noexcept -> void;
            auto set_output_function(std::function<void(output_type)> output) noexcept -> void;

        private:
            auto safe_push(input_type) -> void;
            auto safe_pop(int) -> input_type;
            auto create_filter(int) -> void;
            auto process(int) -> void;

        private:
            std::function<input_type(void)> input_;
            std::function<void(output_type)> output_;

            int devices_;

            std::size_t filter_size_;
            std::size_t n_col_;
            float tau_;
            std::map<int, ddrf::cuda::device_ptr<float>> rs_;

            std::vector<std::queue<input_type>> input_vec_;
            using iv_size_type = typename decltype(input_vec_)::size_type;
    };
}

#endif /* DDAFA_FILTER_STAGE_H_ */
