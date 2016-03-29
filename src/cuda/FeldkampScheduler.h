#ifndef CUDA_FELDKAMPSCHEDULER_H_
#define CUDA_FELDKAMPSCHEDULER_H_

#include <cstddef>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include <boost/log/trivial.hpp>

#include <ddrf/cuda/Check.h>

#include "../common/Geometry.h"


namespace ddafa
{
	namespace cuda
	{
		// FIXME: Make me a "Modern C++ Design"-Singleton
		template <typename T>
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

				static auto instance(const common::Geometry& geo) -> FeldkampScheduler<T>&
				{
					static FeldkampScheduler<T> instance(std::forward<const common::Geometry>(geo));
					return instance;
				}

				auto get_chunks() const noexcept -> std::map<int, std::vector<std::pair<std::uint32_t, std::uint32_t>>>
				{
					return chunks_;
				}

				auto chunkNumber(int device) -> std::uint32_t
				{
					return volumes_per_device_.at(device);
				}

				auto chunkDimensions(int device)
					-> std::vector<std::pair<std::uint32_t, std::uint32_t>>
				{
					return chunks_.at(device);
				}

				auto volumes_per_device() const noexcept -> std::map<int, std::uint32_t>
				{
					return volumes_per_device_;
				}

				auto volume_geometry() const noexcept -> VolumeGeometry
				{
					return vol_geo_;
				}

			protected:
				FeldkampScheduler(const common::Geometry& geo)
				: vol_geo_{0}
				{
					calculate_volume_geo(geo);
					calculate_volume_height_mm();
					auto dist_sd = geo.dist_det + geo.dist_src;

					// calculate volume size in bytes (GPU RAM) and split it up if it doesn't fit
					volume_size_ = vol_geo_.dim_x * vol_geo_.dim_y * vol_geo_.dim_z * sizeof(T);
					auto vol_count = 0u;
					BOOST_LOG_TRIVIAL(debug) << "Volume needs " << volume_size_ << " Bytes.";
					auto dev_count = int{};
					CHECK(cudaGetDeviceCount(&dev_count));
					volume_size_ /= static_cast<unsigned int>(dev_count);
					for(auto i = 0; i <= (dev_count - 1); ++i)
					{
						auto vol_size_dev = volume_size_;
						auto vol_count_dev = 1u;
						CHECK(cudaSetDevice(i));
						auto properties = cudaDeviceProp{};
						CHECK(cudaGetDeviceProperties(&properties, i));

						// divide size by 2 until it fits in memory
						auto calcVolumeSizePerDev = std::function<std::size_t(std::size_t, std::uint32_t*, std::size_t)>();
						calcVolumeSizePerDev = [&calcVolumeSizePerDev](std::size_t volume_size, std::uint32_t* volume_count, std::size_t dev_mem)
						{
							if(volume_size >= dev_mem)
							{
								volume_size /= 2;
								*volume_count *= 2;
								return calcVolumeSizePerDev(volume_size, volume_count, dev_mem);
							}
							else
								return volume_size;
						};

						vol_size_dev = calcVolumeSizePerDev(vol_size_dev, &vol_count_dev, properties.totalGlobalMem);
						vol_count += vol_count_dev;
						auto chunk_str = std::string(vol_count_dev > 1 ? "chunks" : "chunk");
						BOOST_LOG_TRIVIAL(debug) << "Need " << vol_count_dev << " " << chunk_str << " with " << vol_size_dev
							<< " Bytes on device #" << i;
						volumes_per_device_.emplace(std::make_pair(i, vol_count_dev));
					}

					// calculate chunk borders
					auto chunks = std::vector<std::pair<std::uint32_t, std::uint32_t>>{};
					auto first_row = 0.f;
					for(auto n = 0u; n < vol_count; ++n)
					{
						auto top = volume_height_ * ((1.f / 2.f) - (static_cast<float>(n) / vol_count));
						auto bottom = volume_height_ * ((1.f / 2.f) - (static_cast<float>(n + 1) / vol_count));

						auto top_proj = top * (dist_sd / geo.dist_src) + geo.det_offset_vert;
						auto bottom_proj = bottom * (dist_sd / geo.dist_src) + geo.det_offset_vert;

						/*auto top_proj = 0.f;
						auto bottom_proj = 0.f;
						if(n < (vol_count / 2))
						{
							top_proj = (dist_sd * top) / (geo.dist_src - (std::sqrt(2.f) / 2) * volume_height_) + geo.det_offset_vert;
							bottom_proj = (dist_sd * bottom) / (geo.dist_src + (std::sqrt(2.f) / 2) * volume_height_) + geo.det_offset_vert;
						}
						else
						{
							top_proj = (dist_sd * top) / (geo.dist_src + (std::sqrt(2.f) / 2) * volume_height_) + geo.det_offset_vert;
							bottom_proj = (dist_sd * bottom) / (geo.dist_src - (std::sqrt(2.f) / 2) * volume_height_) + geo.det_offset_vert;
						}*/
						// FIXME: This is quite error-prone with regard to non-standard projection sizes (e.g. 401 in y direction)
						auto top_row = std::ceil(top_proj / geo.det_pixel_size_vert);
						auto bottom_row = std::ceil(bottom_proj / geo.det_pixel_size_vert);
						if(n == 0)
							first_row = top_row;

						// normalize so row 0 is actually at row 0 and not in the middle
						auto start_row = std::abs(top_row - first_row);
						auto row_num = std::abs(bottom_row - top_row);
						chunks.emplace_back(std::make_pair(start_row, start_row + row_num - 1));

						BOOST_LOG_TRIVIAL(debug) << "Chunk #" << n << " reaches from " << start_row << " to " << start_row + row_num - 1
								<< ", number of rows: " << row_num;
					}

					// distribute chunks among the devices
					auto chunks_begin = std::begin(chunks);
					for(auto i = 0; i < dev_count; ++i)
					{
						auto chunk_count = volumes_per_device_.at(i);
						chunks_.emplace(std::make_pair(i,
							std::vector<std::pair<std::uint32_t, std::uint32_t>>(chunks_begin, chunks_begin + chunk_count)));
						chunks_begin += chunk_count;

						BOOST_LOG_TRIVIAL(debug) << "Device #" << i << " will process the following chunk(s):";
						auto vec = chunks_.at(i);
						for(auto& p : vec)
							BOOST_LOG_TRIVIAL(debug) << "(" << p.first << "," << p.second << ")";
					}
				}

			private:
				auto calculate_volume_geo(const common::Geometry& geo) noexcept -> void
				{
					// calculate volume dimensions -- x and y
					auto dist_sd = std::abs(geo.dist_det) + std::abs(geo.dist_src);
					auto N_h = geo.det_pixels_row;
					auto d_h = geo.det_pixel_size_horiz;
					auto delta_h = geo.det_offset_horiz;
					auto alpha = std::atan((((N_h * d_h) / 2.f) + std::abs(delta_h)) / dist_sd);
					auto r = std::abs(geo.dist_src) * std::sin(alpha);
					vol_geo_.voxel_size_x = r / ((((N_h * d_h) / 2.f) + std::abs(delta_h)) / d_h);
					vol_geo_.voxel_size_y = vol_geo_.voxel_size_x;
					vol_geo_.dim_x = static_cast<std::size_t>((2.f * r) / vol_geo_.voxel_size_x);
					vol_geo_.dim_y = vol_geo_.dim_x;

					// calculate volume dimensions -- z
					vol_geo_.voxel_size_z = vol_geo_.voxel_size_x;
					auto N_v = geo.det_pixels_column;
					auto d_v = geo.det_pixel_size_vert;
					auto delta_v = geo.det_offset_vert;
					vol_geo_.dim_z = static_cast<std::size_t>(((N_v * d_v) / 2.f + std::abs(delta_v)) * (std::abs(geo.dist_src) / dist_sd) * (2.f / vol_geo_.voxel_size_z));

					BOOST_LOG_TRIVIAL(debug) << "Volume dimensions: " << vol_geo_.dim_x << "x" << vol_geo_.dim_y << "x" << vol_geo_.dim_z;
					BOOST_LOG_TRIVIAL(debug) << "Voxel size: " << vol_geo_.voxel_size_x << "x" << vol_geo_.voxel_size_y << "x" << vol_geo_.voxel_size_z;
				}

				auto calculate_volume_height_mm() -> void
				{
					volume_height_ = vol_geo_.dim_z * vol_geo_.voxel_size_z;
					BOOST_LOG_TRIVIAL(debug) << "Volume is " << volume_height_ << " mm high.";
				}

			private:
				std::map<int, std::vector<std::pair<std::uint32_t, std::uint32_t>>> chunks_;
				std::map<int, std::uint32_t> volumes_per_device_;
				float volume_height_;
				std::size_t volume_size_;
				VolumeGeometry vol_geo_;
		};
	}
}



#endif /* CUDA_FELDKAMPSCHEDULER_H_ */
