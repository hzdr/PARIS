#ifndef DDAFA_FILTER_STAGE_H_
#define DDAFA_FILTER_STAGE_H_

#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <queue>
#include <utility>

#include <ddrf/cuda/memory.h>
#include <ddrf/memory.h>

#include "metadata.h"

namespace ddafa
{
    class filter_stage
    {
        private:
            using device_allocator = ddrf::cuda::device_allocator<float, ddrf::memory_layout::pointer_2D>;
            using pool_allocator = ddrf::pool_allocator<float, ddrf::memory_layout::pointer_2D, device_allocator>;
            using smart_pointer = typename pool_allocator::smart_pointer;

        public:
            using input_type = std::pair<smart_pointer, projection_metadata>;
            using output_type = std::pair<smart_pointer, projection_metadata>;

        public:
            filter_stage(std::uint32_t n_row, std::uint32_t n_col, float l_px_row);

            auto run() -> void;
            auto set_input_function(std::function<input_type(void)> input) noexcept -> void;
            auto set_output_function(std::function<void(output_type)> output) noexcept -> void;

        private:
            auto safe_push(input_type) -> void;
            auto safe_pop() -> input_type;
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

            std::map<int, std::queue<input_type>> input_map_;
            std::atomic_flag lock_;
    };
}

#endif /* DDAFA_FILTER_STAGE_H_ */
