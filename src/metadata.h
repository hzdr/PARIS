#ifndef DDAFA_METADATA_H_
#define DDAFA_METADATA_H_

#include <cstddef>
#include <cstdint>

namespace ddafa
{
    struct projection_metadata
    {
        std::size_t width;
        std::size_t height;
        std::uint64_t index;
        float phi;
        bool valid = false;
        int device;
    };

    struct volume_metadata
    {
        std::size_t width;
        std::size_t height;
        std::size_t depth;
        bool valid = false;
        int device;
    };
}

#endif /* DDAFA_METADATA_H_ */
