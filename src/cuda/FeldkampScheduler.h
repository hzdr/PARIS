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

#define BOOST_ALL_DYN_LINK
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

			protected:
				FeldkampScheduler(const common::Geometry& geo)
				{
					// calculate volume height in mm
					auto dist_src_det = geo.dist_src + geo.dist_det;
					auto det_height = geo.det_pixel_size_vert * geo.det_pixels_column;
					BOOST_LOG_TRIVIAL(debug) << "Detector is " << det_height << " mm high.";
					volume_height_ = (geo.dist_src * det_height) /
										(dist_src_det + (std::sqrt(2.f) / 2) * det_height);
					BOOST_LOG_TRIVIAL(debug) << "Volume is " << volume_height_ << " mm high.";

					// calculate volume size in bytes (GPU RAM) and split it up if it doesn't fit
					volume_size_ = geo.det_pixels_row * geo.det_pixels_row * geo.det_pixels_column * sizeof(T);
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

						auto top_proj = 0.f;
						auto bottom_proj = 0.f;
						if(n < (vol_count / 2))
						{
							top_proj = (dist_src_det * top) / (geo.dist_src - (std::sqrt(2.f) / 2) * volume_height_);
							bottom_proj = (dist_src_det * bottom) / (geo.dist_src + (std::sqrt(2.f) / 2) * volume_height_);
						}
						else
						{
							top_proj = (dist_src_det * top) / (geo.dist_src + (std::sqrt(2.f) / 2) * volume_height_);
							bottom_proj = (dist_src_det * bottom) / (geo.dist_src - (std::sqrt(2.f) / 2) * volume_height_);
						}
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
				std::map<int, std::vector<std::pair<std::uint32_t, std::uint32_t>>> chunks_;
				std::map<int, std::uint32_t> volumes_per_device_;
				float volume_height_;
				std::size_t volume_size_;
		};
	}
}



#endif /* CUDA_FELDKAMPSCHEDULER_H_ */
