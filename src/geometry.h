#ifndef DDAFA_GEOMETRY_H_
#define DDAFA_GEOMETRY_H_

#include <cstdint>

namespace ddafa
{
    struct geometry
    {
        // Detector
        std::uint32_t n_row;    // pixels per row
        std::uint32_t n_col;    // pixels per column
        float l_px_row;         // pixel size in horizontal direction [mm]
        float l_px_col;         // pixel size in vertical direction [mm]
        float delta_s;          // offset in horizontal direction [px]
        float delta_t;          // offset in vertical direction [px]

        // Support
        float d_so;             // distance source->object
        float d_od;             // distance object->detector

        // Rotation
        float delta_phi;        // difference between two successive angles - ignored if there is an angle file
    };
}

#endif /* GEOMETRY_H_ */
