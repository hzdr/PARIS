#ifndef DDAFA_GEOMETRY_CALCULATOR_H_
#define DDAFA_GEOMETRY_CALCULATOR_H_

#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

#include "geometry.h"
#include "metadata.h"

namespace ddafa
{
    class geometry_calculator
    {
        public:
            geometry_calculator(const geometry& geo);

            auto get_projection_iteration_num() const noexcept -> std::uint32_t;
            auto get_subvolume_metadata() const noexcept -> std::vector<volume_metadata>;

        private:
            auto calculate_volume_width_height_vx() noexcept -> void;
            auto calculate_volume_depth_vx() noexcept -> void;
            auto calculate_volume_height_mm() noexcept -> void;
            auto calculate_memory_footprint() noexcept -> void;
            auto calculate_volume_partition() -> void;
            auto calculate_subvolume_offsets() noexcept -> void;

        private:
            geometry det_geo_;
            volume_metadata vol_geo_;
            float d_sd_;
            int devices_;
            float vol_height_;
            std::size_t vol_mem_;
            std::size_t proj_mem_;
            std::uint32_t vol_count_;
            std::map<int, std::uint32_t> vol_per_dev_;
            std::map<int, volume_metadata> vol_geo_per_dev_;
    };
}



#endif /* DDAFA_GEOMETRY_CALCULATOR_H_ */
