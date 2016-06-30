#ifndef CUDA_WEIGHTING_H_
#define CUDA_WEIGHTING_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <map>
#include <thread>

#include <ddrf/Queue.h>
#include <ddrf/Image.h>
#include <ddrf/cuda/DeviceMemoryManager.h>
#include <ddrf/cuda/HostMemoryManager.h>
#include <ddrf/default/MemoryManager.h>

#include "../common/Geometry.h"

namespace ddafa
{
    namespace cuda
    {
        class Weighting
        {
            public:
                using input_type = ddrf::Image<ddrf::cuda::DeviceMemoryManager<float>>;
                using output_type = ddrf::Image<ddrf::cuda::DeviceMemoryManager<float>>;

            public:
                Weighting(const common::Geometry& geo);
                auto process(input_type&& img) -> void;
                auto wait() -> output_type;
                auto set_input_num(std::uint32_t num) -> void;

            protected:
                ~Weighting() = default;

            private:
                auto processor(int) -> void;

            private:
                ddafa::common::Geometry geo_;
                ddafa::common::Geometry old_geo_; // we need this to ensure that we always get the same scheduler instance
                std::map<int, ddrf::Queue<input_type>> map_imgs_;
                ddrf::Queue<output_type> results_;
                float h_min_;
                float v_min_;
                float d_dist_;
                int devices_;

                std::map<int, std::thread> processor_threads_;

                std::uint32_t input_num_;
                std::atomic_bool input_num_set_;
        };
    }
}


#endif /* CUDAWEIGHTING_H_ */
