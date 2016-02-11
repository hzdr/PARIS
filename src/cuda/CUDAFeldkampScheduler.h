/*
 * CUDAFeldkampScheduler.h
 *
 *  Created on: 04.02.2016
 *      Author: Jan Stephan
 */

#ifndef CUDAFELDKAMPSCHEDULER_H_
#define CUDAFELDKAMPSCHEDULER_H_

#include <cstddef>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#define BOOST_ALL_DYN_LINK
#include <boost/log/trivial.hpp>

#include "../common/Geometry.h"
#include "CUDAAssert.h"

namespace ddafa
{
	namespace impl
	{
		// FIXME: Make me a "Modern C++ Design"-Singleton
		template <typename T>
		class CUDAFeldkampScheduler
		{
			public:
				~CUDAFeldkampScheduler()
				{

				}

				static CUDAFeldkampScheduler<T>& instance(const common::Geometry& geo)
				{
					static CUDAFeldkampScheduler<T> instance(std::forward<const common::Geometry>(geo));
					return instance;
				}

				std::uint32_t chunkNumber(int device)
				{
					return volumes_per_device_.at(device);
				}

				std::vector<std::pair<std::uint32_t, std::uint32_t>> chunkDimensions(int device)
				{
					return chunks_.at(device);
				}

			protected:
				CUDAFeldkampScheduler(const common::Geometry& geo)
				{
					// calculate volume height in mm
					float dist_src_det = geo.dist_src + geo.dist_det;
					float det_height = geo.det_pixel_size_vert * geo.det_pixels_column;
					BOOST_LOG_TRIVIAL(debug) << "Detector is " << det_height << " mm high.";
					volume_height_ = (geo.dist_src * det_height) /
										(dist_src_det + (std::sqrt(2.f) / 2) * det_height);
					BOOST_LOG_TRIVIAL(debug) << "Volume is " << volume_height_ << " mm high.";

					// calculate volume size in bytes (GPU RAM) and split it up if it doesn't fit
					volume_size_ = geo.det_pixels_row * geo.det_pixels_row * geo.det_pixels_column * sizeof(T);
					std::uint32_t volume_count = 0;
					BOOST_LOG_TRIVIAL(debug) << "Volume needs " << volume_size_ << " Bytes.";
					int device_count;
					assertCuda(cudaGetDeviceCount(&device_count));
					volume_size_ /= device_count;
					for(int i = 0; i <= (device_count - 1); ++i)
					{
						std::size_t vol_size_dev = volume_size_;
						std::uint32_t vol_count_dev = 1;
						assertCuda(cudaSetDevice(i));
						cudaDeviceProp properties;
						assertCuda(cudaGetDeviceProperties(&properties, i));

						// divide size by 2 until it fits onto memory
						vol_size_dev = calcVolumeSizePerDev(vol_size_dev, &vol_count_dev, properties.totalGlobalMem);
						volume_count += vol_count_dev;
						std::string chunk_str = vol_count_dev > 1 ? "chunks" : "chunk";
						BOOST_LOG_TRIVIAL(debug) << "Need " << vol_count_dev << " " << chunk_str << " with " << vol_size_dev
							<< " Bytes on device #" << i;
						volumes_per_device_.emplace(std::make_pair(i, vol_count_dev));
					}

					// calculate chunk borders
					std::vector<std::pair<std::uint32_t, std::uint32_t>> chunks;
					std::int32_t first_row;
					for(std::uint32_t n = 0; n < volume_count; ++n)
					{
						float top = volume_height_ * ((1.f / 2.f) - (static_cast<float>(n) / volume_count));
						float bottom = volume_height_ * ((1.f / 2.f) - (static_cast<float>(n + 1) / volume_count));

						float top_proj, bottom_proj;
						if(n < (volume_count / 2))
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
						std::int32_t top_row = std::ceil(top_proj / geo.det_pixel_size_vert);
						std::int32_t bottom_row = std::ceil(bottom_proj / geo.det_pixel_size_vert);
						if(n == 0)
							first_row = top_row;

						// normalize so row 0 is actually at row 0 and not in the middle
						std::uint32_t start_row = std::abs(top_row - first_row);
						std::uint32_t row_num = std::abs(bottom_row - top_row);
						chunks.emplace_back(std::make_pair(start_row, start_row + row_num - 1));

						BOOST_LOG_TRIVIAL(debug) << "Chunk #" << n << " reaches from " << start_row << " to " << start_row + row_num - 1
								<< ", number of rows: " << row_num;
					}

					// distribute chunks among the devices
					auto chunks_begin = std::begin(chunks);
					for(int i = 0; i < device_count; ++i)
					{
						std::uint32_t chunk_count = volumes_per_device_.at(i);
						chunks_.emplace(std::make_pair(i,
							std::vector<std::pair<std::uint32_t, std::uint32_t>>(chunks_begin, chunks_begin + chunk_count)));
						chunks_begin += chunk_count;

						BOOST_LOG_TRIVIAL(debug) << "Device #" << i << " will process the following chunk(s):";
						auto vec = chunks_.at(i);
						for(auto p : vec)
							BOOST_LOG_TRIVIAL(debug) << "(" << p.first << "," << p.second << ")";
					}
				}

			private:
				std::size_t calcVolumeSizePerDev(std::size_t volume_size, std::uint32_t* volume_count, std::size_t dev_mem)
				{
					if(volume_size >= dev_mem)
					{
						volume_size /= 2;
						*volume_count *= 2;
						return calcVolumeSizePerDev(volume_size, volume_count, dev_mem);
					}
					else
						return volume_size;
				}

			private:
				float volume_height_;
				std::size_t volume_size_;
				std::map<int, std::vector<std::pair<std::uint32_t, std::uint32_t>>> chunks_;
				std::map<int, std::uint32_t> volumes_per_device_;
		};
	}
}



#endif /* CUDAFELDKAMPSCHEDULER_H_ */
