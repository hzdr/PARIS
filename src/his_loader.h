#ifndef DDAFA_HIS_LOADER_H_
#define DDAFA_HIS_LOADER_H_

#include <string>
#include <utility>
#include <vector>

#include <ddrf/cuda/memory.h>
#include <ddrf/memory.h>

#include "metadata.h"

namespace ddafa
{
    class his_loader
    {
        public:
            using cuda_host_allocator = ddrf::cuda::host_allocator<float, ddrf::memory_layout::pointer_2D>;
            using pool_allocator = ddrf::pool_allocator<float, ddrf::memory_layout::pointer_2D, cuda_host_allocator>;
            using smart_pointer = typename pool_allocator::smart_pointer;
            using image_type = std::pair<smart_pointer, projection_metadata>;

            his_loader() noexcept = default;
            auto load(const std::string& path) -> std::vector<image_type>;

        private:
             pool_allocator alloc_;
    };
}



#endif /* DDAFA_HIS_LOADER_H_ */
