#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <iomanip>
#include <iterator>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

#include <boost/log/trivial.hpp>

#include <ddrf/cuda/Check.h>
#include <ddrf/cuda/Memory.h>

#include "../common/Geometry.h"
#include "FeldkampScheduler.h"

namespace ddafa
{
    namespace cuda
    {
        FeldkampScheduler::FeldkampScheduler(const common::Geometry& geo, volume_type vol_type)
        : det_geo_(geo), vol_geo_{0}
        , proj_add_top_{0} , proj_add_bot_{0}
        , volume_count_{0u}, dist_sd_{std::abs(det_geo_.dist_det) + std::abs(det_geo_.dist_src)}
        {
            CHECK(cudaGetDeviceCount(&devices_));
            for(auto d = 0; d < devices_; ++d)
            {
                proj_counters_[d] = 0;
                // the following constructs a mutex and a condition_variable in place
                pc_mutexes_[d];
                pc_cvs_[d];
            }

            apply_offset_to_projection();
            calculate_volume_width_height_vx();
            calculate_volume_slices_vx();
            calculate_volume_height_mm();
            calculate_volume_bytes(vol_type);
            calculate_volumes_per_device(vol_type);
            calculate_subvolume_offsets();
            calculate_subprojection_borders();
            distribute_subprojections();
            calculate_subprojection_offsets();
        }

        auto FeldkampScheduler::instance(const common::Geometry& geo, volume_type vol_type) -> FeldkampScheduler&
        {
            static FeldkampScheduler instance{geo, vol_type};
            return instance;
        }

        auto FeldkampScheduler::get_volume_num(int device) const noexcept -> std::uint32_t
        {
            try
            {
                return volumes_per_device_.at(device);
            }
            catch(const std::out_of_range&)
            {
                return 0u;
            }
        }

        auto FeldkampScheduler::get_volume_offset(int device, std::uint32_t index) const noexcept -> std::size_t
        {
            try
            {
                auto offset_map = offset_per_volume_.at(device);
                return offset_map.at(index);
            }
            catch(const std::out_of_range&)
            {
                return static_cast<std::size_t>(0);
            }
        }

        auto FeldkampScheduler::get_subproj_num(int device) const noexcept -> std::uint32_t
        {
            return get_volume_num(device);
        }

        auto FeldkampScheduler::get_subproj_dims(int device, std::size_t index) const noexcept -> std::pair<std::uint32_t, std::uint32_t>
        {
            try
            {
                auto vec = subprojs_.at(device);
                return vec.at(index);
            }
            catch(const std::out_of_range&)
            {
                return std::make_pair<std::uint32_t, std::uint32_t>(0u, 0u);
            }
        }

        auto FeldkampScheduler::get_subproj_offset(int device, std::uint32_t index) const noexcept -> std::size_t
        {
            try
            {
                auto offset_map = offset_per_subproj_.at(device);
                return offset_map.at(index);
            }
            catch(const std::out_of_range&)
            {
                return static_cast<std::size_t>(0);
            }
        }

        auto FeldkampScheduler::get_volume_geometry() const noexcept -> VolumeGeometry
        {
            return vol_geo_;
        }

        auto FeldkampScheduler::get_additional_proj_lines_top() const noexcept -> std::size_t
        {
            return proj_add_top_;
        }

        auto FeldkampScheduler::get_additional_proj_lines_bot() const noexcept -> std::size_t
        {
            return proj_add_bot_;
        }

        auto FeldkampScheduler::get_updated_detector_geometry() const noexcept -> common::Geometry
        {
            return det_geo_;
        }

        auto FeldkampScheduler::acquire_projection(int device) noexcept -> void
        {
            BOOST_LOG_TRIVIAL(debug) << "Acquiring projection on device #" << device;
            try
            {
                auto lock = std::unique_lock<std::mutex>{pc_mutexes_.at(device)};
                while(proj_counters_.at(device) >= 30)
                    pc_cvs_.at(device).wait(lock);

                ++(proj_counters_.at(device));
            }
            catch(const std::out_of_range&)
            {
                BOOST_LOG_TRIVIAL(error) << "cuda::FeldkampScheduler: Invalid device specified";
                std::terminate();
            }
        }

        auto FeldkampScheduler::release_projection(int device) noexcept -> void
        {
            BOOST_LOG_TRIVIAL(debug) << "Releasing projection on device #" << device;
            try
            {
                auto lock = std::unique_lock<std::mutex>{pc_mutexes_.at(device)};
                if(proj_counters_.at(device) > 0)
                    --(proj_counters_.at(device));
                pc_cvs_.at(device).notify_one();
            }
            catch(const std::out_of_range&)
            {
                BOOST_LOG_TRIVIAL(error) << "cuda::FeldkampScheduler: Invalid device specified: " << device;
                std::terminate();
            }
        }

        auto FeldkampScheduler::apply_offset_to_projection() -> void
        {
            auto proj_height_px_f = static_cast<float>(det_geo_.det_pixels_column);
            auto pixel_size_mm = det_geo_.det_pixel_size_vert;
            auto proj_height_mm = proj_height_px_f * pixel_size_mm;
            auto vert_offset_mm = det_geo_.det_offset_vert * pixel_size_mm;

            BOOST_LOG_TRIVIAL(info) << "Applying offset of " << vert_offset_mm << "mm";
            BOOST_LOG_TRIVIAL(info) << "Old projection height: " << proj_height_mm << "mm";
            proj_height_mm += std::abs(vert_offset_mm);
            if(vert_offset_mm < 0.f)
                proj_add_top_ += static_cast<std::uint32_t>(std::abs(vert_offset_mm) / pixel_size_mm);
            else
                proj_add_bot_ += static_cast<std::uint32_t>(vert_offset_mm / pixel_size_mm);
            BOOST_LOG_TRIVIAL(info) << "New projection height: " << proj_height_mm << "mm";
            proj_height_px_f = proj_height_mm / pixel_size_mm;

            auto proj_height_px = static_cast<std::uint32_t>(proj_height_px_f);
            BOOST_LOG_TRIVIAL(info) << "Projection height expanded to " << proj_height_px << "px, was: "
                    << det_geo_.det_pixels_column << "px";

            det_geo_.det_pixels_column = proj_height_px;
        }

        auto FeldkampScheduler::calculate_volume_width_height_vx() -> void
        {
            auto N_h = static_cast<float>(det_geo_.det_pixels_row);
            auto d_h = det_geo_.det_pixel_size_horiz;
            auto delta_h = det_geo_.det_offset_horiz * d_h; // the offset is measured in pixels!
            auto alpha = std::atan((((N_h * d_h) / 2.f) + std::abs(delta_h)) / dist_sd_);
            auto r = std::abs(det_geo_.dist_src) * std::sin(alpha);
            vol_geo_.voxel_size_x = r / ((((N_h * d_h) / 2.f) + std::abs(delta_h)) / d_h);
            vol_geo_.voxel_size_y = vol_geo_.voxel_size_x;
            vol_geo_.dim_x = static_cast<std::size_t>((2.f * r) / vol_geo_.voxel_size_x);
            vol_geo_.dim_y = vol_geo_.dim_x;
        }

        /*
         * This method expands the projection height according to the following algorithm:
         *      1) Calculate the number of slices in the output volume
         *      2) If slices mod 64 != 0, append an additional line to the bottom of the projection and recalculate the number of slices
         *      3) If slices mod 64 != 0, put an additional line on the top of the projection and recalculate the number of slices
         *      4) If slices mod 64 != 0, go to 2), else go to 5)
         *      5) The volume is now divisible into 64 subvolumes which should be enough for any hardware that is CUDA enabled
         */
        auto FeldkampScheduler::calculate_volume_slices_vx() -> void
        {
            vol_geo_.voxel_size_z = vol_geo_.voxel_size_x;
            auto N_v = static_cast<float>(det_geo_.det_pixels_column);
            auto d_v = det_geo_.det_pixel_size_vert;
            auto delta_v = det_geo_.det_offset_vert * d_v;

            auto calc_dim_z = [&]() {
                return static_cast<std::size_t>(((N_v * d_v) / 2.f + std::abs(delta_v)) * (std::abs(det_geo_.dist_src) / dist_sd_) * (2.f / vol_geo_.voxel_size_z));
            };


            vol_geo_.dim_z = calc_dim_z();

            auto add_to_bot = true;
            while(vol_geo_.dim_z % 64 != 0)
            {
                ++N_v;
                add_to_bot ? ++proj_add_bot_ : ++proj_add_top_;
                add_to_bot = !add_to_bot;
                vol_geo_.dim_z = calc_dim_z();
            }
            det_geo_.det_pixels_column = static_cast<std::uint32_t>(N_v);

            BOOST_LOG_TRIVIAL(info) << "Requires additional projection lines: " << proj_add_top_ << " to the top and " << proj_add_bot_ << " to the bottom";

            BOOST_LOG_TRIVIAL(info) << "Volume dimensions: " << vol_geo_.dim_x << " x " << vol_geo_.dim_y << " x " << vol_geo_.dim_z << " vx";
                        BOOST_LOG_TRIVIAL(info) << "Voxel size: " << std::setprecision(4) << vol_geo_.voxel_size_x << " x " << vol_geo_.voxel_size_y << " x " << vol_geo_.voxel_size_z << " mm³";

            BOOST_LOG_TRIVIAL(info) << "Working projection dimensions: " << det_geo_.det_pixels_row << " x " << det_geo_.det_pixels_column << " px²";
        }

        auto FeldkampScheduler::calculate_volume_height_mm() -> void
        {
            volume_height_ = static_cast<float>(vol_geo_.dim_z) * vol_geo_.voxel_size_z;
            BOOST_LOG_TRIVIAL(info) << "Volume is " << std::setprecision(4) << volume_height_ << "mm high.";
        }

        auto FeldkampScheduler::calculate_volume_bytes(volume_type vol_type) -> void
        {
            switch(vol_type)
            {
                case volume_type::int8:
                    type_bytes_ = sizeof(std::int8_t);
                    break;

                case volume_type::uint8:
                    type_bytes_ = sizeof(std::uint8_t);
                    break;

                case volume_type::int16:
                    type_bytes_ = sizeof(std::int16_t);
                    break;

                case volume_type::uint16:
                    type_bytes_ = sizeof(std::uint16_t);
                    break;

                case volume_type::int32:
                    type_bytes_ = sizeof(std::int32_t);
                    break;

                case volume_type::uint32:
                    type_bytes_ = sizeof(std::uint32_t);
                    break;

                case volume_type::int64:
                    type_bytes_ = sizeof(std::int64_t);
                    break;

                case volume_type::uint64:
                    type_bytes_ = sizeof(std::uint64_t);
                    break;

                case volume_type::single_float:
                    type_bytes_ = sizeof(float);
                    break;

                case volume_type::double_float:
                    type_bytes_ = sizeof(double);
                    break;

                // CUDA currently (v7.5) doesn't support long double and shortens the type to double
                case volume_type::long_double_float:
                    type_bytes_ = sizeof(double);
                    break;
            }

            volume_bytes_ = vol_geo_.dim_x * vol_geo_.dim_y * vol_geo_.dim_z * type_bytes_;
            BOOST_LOG_TRIVIAL(info) << "Volume requires " << volume_bytes_ << " bytes.";

            projection_bytes_ = det_geo_.det_pixels_row * det_geo_.det_pixels_column * type_bytes_;
            BOOST_LOG_TRIVIAL(info) << "One projection requires " << projection_bytes_ << " bytes.";
        }

        auto FeldkampScheduler::calculate_volumes_per_device(volume_type vol_type) -> void
        {
            // split volume up if it doesn't fit into device memory
            volume_bytes_ /= static_cast<unsigned int>(devices_);
            projection_bytes_ /= static_cast<unsigned int>(devices_);
            for(auto i = 0; i < devices_; ++i)
            {
                auto required_mem = volume_bytes_ + 32 * projection_bytes_;
                auto vol_count_dev = 1u;
                CHECK(cudaSetDevice(i));
                auto free_mem = std::size_t{};
                auto total_mem = std::size_t{};
                auto max_malloc = std::size_t{};
                CHECK(cudaMemGetInfo(&free_mem, &total_mem));
                CHECK(cudaDeviceGetLimit(&max_malloc, cudaLimitMallocHeapSize));

                // divide volume size by 2 until it and (roughly) 32 projections fit into memory
                auto calcVolumeSizePerDev = std::function<std::size_t(std::size_t, std::size_t, std::size_t, std::uint32_t*, std::size_t)>();
                calcVolumeSizePerDev = [&calcVolumeSizePerDev](std::size_t mem_required, std::size_t volume_size, std::size_t proj_size,
                                                                std::uint32_t* volume_count, std::size_t dev_mem)
                {
                    if(mem_required >= dev_mem)
                    {
                        volume_size /= 2;
                        proj_size /= 2;
                        *volume_count *= 2;
                        mem_required = volume_size + 32 * proj_size;
                        return calcVolumeSizePerDev(mem_required, volume_size, proj_size, volume_count, dev_mem);
                    }
                    else
                        return mem_required;
                };

                required_mem = calcVolumeSizePerDev(required_mem, volume_bytes_, projection_bytes_, &vol_count_dev, free_mem);

                try
                {
                    BOOST_LOG_TRIVIAL(info) << "Trying test allocation...";
                    auto d = static_cast<std::size_t>(devices_);
                    //FIXME: This is really ugly
                    switch(vol_type)
                    {
                        case volume_type::int8:
                            ddrf::cuda::make_device_ptr<std::int8_t>(vol_geo_.dim_x, vol_geo_.dim_y, ((vol_geo_.dim_z / d) / vol_count_dev));
                            break;

                        case volume_type::uint8:
                            ddrf::cuda::make_device_ptr<std::uint8_t>(vol_geo_.dim_x, vol_geo_.dim_y, ((vol_geo_.dim_z / d) / vol_count_dev));
                            break;

                        case volume_type::int16:
                            ddrf::cuda::make_device_ptr<std::int16_t>(vol_geo_.dim_x, vol_geo_.dim_y, ((vol_geo_.dim_z / d) / vol_count_dev));
                            break;

                        case volume_type::uint16:
                            ddrf::cuda::make_device_ptr<std::uint16_t>(vol_geo_.dim_x, vol_geo_.dim_y, ((vol_geo_.dim_z / d) / vol_count_dev));
                            break;

                        case volume_type::int32:
                            ddrf::cuda::make_device_ptr<std::int32_t>(vol_geo_.dim_x, vol_geo_.dim_y, ((vol_geo_.dim_z / d) / vol_count_dev));
                            break;

                        case volume_type::uint32:
                            ddrf::cuda::make_device_ptr<std::uint8_t>(vol_geo_.dim_x, vol_geo_.dim_y, ((vol_geo_.dim_z / d) / vol_count_dev));
                            break;

                        case volume_type::int64:
                            ddrf::cuda::make_device_ptr<std::int64_t>(vol_geo_.dim_x, vol_geo_.dim_y, ((vol_geo_.dim_z / d) / vol_count_dev));
                            break;

                        case volume_type::uint64:
                            ddrf::cuda::make_device_ptr<std::uint64_t>(vol_geo_.dim_x, vol_geo_.dim_y, ((vol_geo_.dim_z / d) / vol_count_dev));
                            break;

                        case volume_type::single_float:
                            ddrf::cuda::make_device_ptr<float>(vol_geo_.dim_x, vol_geo_.dim_y, ((vol_geo_.dim_z / d) / vol_count_dev));
                            break;

                        case volume_type::double_float:
                            ddrf::cuda::make_device_ptr<double>(vol_geo_.dim_x, vol_geo_.dim_y, ((vol_geo_.dim_z / d) / vol_count_dev));
                            break;

                        // CUDA currently (v7.5) doesn't support long double and shortens the type to double
                        case volume_type::long_double_float:
                            ddrf::cuda::make_device_ptr<double>(vol_geo_.dim_x, vol_geo_.dim_y, ((vol_geo_.dim_z / d) / vol_count_dev));
                            break;
                    }
                    BOOST_LOG_TRIVIAL(info) << "Test allocation successful.";
                }
                catch(const ddrf::cuda::out_of_memory&)
                {
                    BOOST_LOG_TRIVIAL(info) << "Test allocation failed, reducing subvolume size.";
                    volume_bytes_ /= 2;
                    projection_bytes_ /= 2;
                    vol_count_dev *= 2;
                    required_mem = volume_bytes_ + 32 * projection_bytes_;
                }

                volume_count_ += vol_count_dev;
                auto chunk_str = std::string{vol_count_dev > 1 ? "chunks" : "chunk"};
                BOOST_LOG_TRIVIAL(info) << "Requires " << vol_count_dev << " " << chunk_str << " with " << required_mem
                    << " bytes on device #" << i;
                volumes_per_device_.emplace(std::make_pair(i, vol_count_dev));
            }
        }

        auto FeldkampScheduler::calculate_subvolume_offsets() -> void
        {
            for(auto i = 0; i < devices_; ++i)
            {
                if(volumes_per_device_.count(i) == 0)
                    continue;

                auto vol_offset = vol_geo_.dim_z / volume_count_;
                auto vol_count_dev = volumes_per_device_[i];

                for(auto c = 0u; c < vol_count_dev; ++c)
                {
                    offset_per_volume_[i][c] = static_cast<std::size_t>(i) * vol_count_dev * vol_offset + c * vol_offset;
                    BOOST_LOG_TRIVIAL(info) << "Subvolume #" << c << " on device #" << i << " has an offset of " << offset_per_volume_[i][c] << "px";
                }

            }
        }

        auto FeldkampScheduler::calculate_subprojection_borders() -> void
        {
            auto delta_v = det_geo_.det_offset_vert * det_geo_.det_pixel_size_vert;
            auto d_v = det_geo_.det_pixel_size_vert;
            auto N_v = static_cast<float>(det_geo_.det_pixels_column);
            auto N = static_cast<float>(volume_count_);
            auto d_src = det_geo_.dist_src;

            auto N_x = static_cast<float>(vol_geo_.dim_x);
            auto d_x = vol_geo_.voxel_size_x;
            auto N_y = static_cast<float>(vol_geo_.dim_y);
            auto d_y = vol_geo_.voxel_size_y;
            auto r_max = std::sqrt(std::pow(N_x, 2.f) * std::pow(d_x, 2.f) + std::pow(N_y, 2.f) * std::pow(d_y, 2.f)) / 2.f;

            for(auto n = 0u; n < volume_count_; ++n)
            {
                auto n_f = static_cast<float>(n);

                auto top = -(volume_height_ / 2.f) + (n_f / N) * volume_height_;
                auto bottom = -(volume_height_ / 2.f) + ((n_f + 1.f) / N) * volume_height_;

                auto top_proj_virt = top * (dist_sd_) / (std::abs(d_src) + (top < 0.f ? -r_max : r_max));
                auto bottom_proj_virt = bottom * (dist_sd_) / (std::abs(d_src) + (bottom < 0.f ? r_max : -r_max));

                auto top_proj_real = 0.f - ((N_v * d_v) / 2.f) - delta_v + (d_v / 2.f);
                auto bottom_proj_real = top_proj_real + (N_v * d_v) - d_v;

                auto top_proj = float{};
                if(top_proj_virt > bottom_proj_real)
                    top_proj = bottom_proj_real;
                else if(top_proj_virt < top_proj_real)
                    top_proj = top_proj_real;
                else
                    top_proj = top_proj_virt;

                auto bottom_proj = float{};
                if(bottom_proj_virt < top_proj_real)
                    bottom_proj = top_proj_real;
                else if(bottom_proj_virt > bottom_proj_real)
                    bottom_proj = bottom_proj_real;
                else
                    bottom_proj = bottom_proj_virt;

                auto start_row = std::floor(((top_proj + ((N_v * d_v) / 2.f) + delta_v) / d_v) - (1.f / 2.f));
                auto bottom_row = std::ceil(((bottom_proj + ((N_v * d_v) / 2.f) + delta_v) / d_v) - (1.f / 2.f));

                if(start_row < 0.f)
                    start_row = 0.f;
                if(bottom_row >= N_v)
                    bottom_row = N_v - 1.f;

                auto start_row_ui = static_cast<std::uint32_t>(start_row);
                auto bottom_row_ui = static_cast<std::uint32_t>(bottom_row);
                subproj_dims_.emplace_back(std::make_pair(start_row_ui, bottom_row_ui));

                BOOST_LOG_TRIVIAL(info) << "For volume #" << n << ": ";
                BOOST_LOG_TRIVIAL(info) << "\t" << "(top-most slice, bottom-most slice) = (" << top << "mm, " << bottom << "mm)";
                BOOST_LOG_TRIVIAL(info) << "\t" << "(top-most virtual projection row, bottom-most virtual projection row) = (" << top_proj_virt << "mm, " << bottom_proj_virt << "mm)";
                BOOST_LOG_TRIVIAL(info) << "\t" << "(top-most actual projection row, bottom-most actual projection row) = (" << top_proj_real << "mm, " << bottom_proj_real << "mm)";
                BOOST_LOG_TRIVIAL(info) << "\t" << "(top-most subprojection row, bottom-most subprojection row) = (" << top_proj << "mm, " << bottom_proj << "mm)";
                BOOST_LOG_TRIVIAL(info) << "\t" << "(top-most subprojection row, bottom-most subprojection row) = (" << start_row << "px, " << bottom_row << "px)";
            }
        }

        auto FeldkampScheduler::distribute_subprojections() -> void
        {
            auto subprojs_begin = std::begin(subproj_dims_);
            for(auto i = 0; i < devices_; ++i)
            {
                auto subprojs_count = get_volume_num(i);
                subprojs_.emplace(std::make_pair(i,
                    std::vector<std::pair<std::uint32_t, std::uint32_t>>(subprojs_begin, subprojs_begin + subprojs_count)));
                subprojs_begin += subprojs_count;

                auto vec = subprojs_.at(i);
                auto subproj_string = vec.size() > 1 ? std::string{"subprojections"} : std::string{"subprojection"};
                BOOST_LOG_TRIVIAL(info) << "Device #" << i << " will process the following " << subproj_string;

                for(auto& p : vec)
                    BOOST_LOG_TRIVIAL(info) << "\t" << "(" << p.first << "px, " << p.second << "px)";
            }
        }

        auto FeldkampScheduler::calculate_subprojection_offsets() -> void
        {
            for(auto i = 0; i < devices_; ++i)
            {
                auto subprojs_count = get_volume_num(i);
                if(subprojs_.count(i) == 0)
                    continue;

                auto vec = subprojs_[i];
                for(auto c = 0u; c < subprojs_count; ++c)
                    offset_per_subproj_[i][c] = vec[c].first;
            }
        }
    }
}

