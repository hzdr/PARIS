#ifndef DDAFA_TIFF_SAVER_H_
#define DDAFA_TIFF_SAVER_H_

#include <string>
#include <utility>

#include <ddrf/cuda/memory.h>

#include "metadata.h"

namespace ddafa
{
    class tiff_saver
    {
        tiff_saver() noexcept = default;
        auto save(std::pair<ddrf::cuda::pinned_host_ptr<float>, volume_metadata> vol, const std::string& path) const -> void;
    };
}



#endif /* TIFF_SAVER_H_ */
