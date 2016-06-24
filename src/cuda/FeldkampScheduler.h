#ifndef CUDA_FELDKAMPSCHEDULER_H_
#define CUDA_FELDKAMPSCHEDULER_H_

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "../common/Geometry.h"


namespace ddafa
{
    namespace cuda
    {
        enum class volume_type
        {
            int8,
            uint8,
            int16,
            uint16,
            int32,
            uint32,
            int64,
            uint64,
            single_float,
            double_float,
            long_double_float
        };

        class FeldkampScheduler
        {
            public:
                struct VolumeGeometry
                {
                    std::size_t dim_x;
                    std::size_t dim_y;
                    std::size_t dim_z;

                    float voxel_size_x;
                    float voxel_size_y;
                    float voxel_size_z;
                };

                ~FeldkampScheduler() = default;

                static auto instance(const common::Geometry& geo, volume_type vol_type) -> FeldkampScheduler&;
                auto get_volume_num(int device) const noexcept -> std::uint32_t;
                auto get_volume_offset(int device, std::uint32_t index) const noexcept -> std::size_t;
                auto get_subproj_num(int device) const noexcept -> std::uint32_t;
                auto get_subproj_dims(int device, std::size_t index) const noexcept -> std::pair<std::uint32_t, std::uint32_t>;
                auto get_subproj_offset(int device, std::uint32_t index) const noexcept -> std::size_t;
                auto get_volume_geometry() const noexcept -> VolumeGeometry;
                auto get_additional_proj_lines_top() const noexcept -> std::size_t;
                auto get_additional_proj_lines_bot() const noexcept -> std::size_t;
                auto get_updated_detector_geometry() const noexcept -> common::Geometry;

                auto acquire_projection(int device) noexcept -> void;
                auto release_projection(int device) noexcept -> void;

                FeldkampScheduler(const FeldkampScheduler&) = delete;
                auto operator=(const FeldkampScheduler&) -> FeldkampScheduler& = delete;
                FeldkampScheduler(FeldkampScheduler&&) = delete;
                auto operator=(FeldkampScheduler&&) -> FeldkampScheduler& = delete;

            protected:
                FeldkampScheduler(const common::Geometry&, volume_type);

            private:
                auto apply_offset_to_projection() -> void;
                auto calculate_volume_width_height_vx() -> void;
                auto calculate_volume_slices_vx() -> void;
                auto calculate_volume_height_mm() -> void;
                auto calculate_volume_bytes(volume_type) -> void;
                auto calculate_volumes_per_device(volume_type) -> void;
                auto calculate_subvolume_offsets() -> void;
                auto calculate_subprojection_borders() -> void;
                auto distribute_subprojections() -> void;
                auto calculate_subprojection_offsets() -> void;

            private:
                common::Geometry det_geo_;
                VolumeGeometry vol_geo_;

                std::size_t proj_add_top_;
                std::size_t proj_add_bot_;

                float volume_height_;
                std::size_t type_bytes_;
                std::size_t volume_bytes_;
                std::size_t projection_bytes_;

                int devices_;
                std::uint32_t volume_count_;
                std::map<int, std::uint32_t> volumes_per_device_;
                std::map<int, std::map<std::uint32_t, std::size_t>> offset_per_volume_;
                float dist_sd_;
                std::vector<std::pair<std::uint32_t, std::uint32_t>> subproj_dims_;
                std::map<int, std::vector<std::pair<std::uint32_t, std::uint32_t>>> subprojs_;
                std::map<int, std::map<std::uint32_t, std::size_t>> offset_per_subproj_;

                std::map<int, int> proj_counters_;
                std::map<int, std::mutex> pc_mutexes_;
                std::map<int, std::condition_variable> pc_cvs_;
        };
    }
}



#endif /* CUDA_FELDKAMPSCHEDULER_H_ */
