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
        std::size_t remainder;
        std::size_t offset;
        bool valid = false;
        int device;

        float vx_size_x;
        float vx_size_y;
        float vx_size_z;
    };
}

#endif /* DDAFA_METADATA_H_ */
