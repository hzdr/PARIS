#include <algorithm>
#include <cstddef>
#include <iterator>
#include <map>
#include <memory>
#include <utility>

#define BOOST_ALL_DYN_LINK
#include <boost/log/trivial.hpp>

#include <ddrf/cuda/Check.h>
#include <ddrf/cuda/Memory.h>

#include "../common/Geometry.h"

#include "FeldkampScheduler.h"
#include "Preloader.h"

namespace ddafa
{
	namespace cuda
	{
		Preloader::Preloader(const common::Geometry& geo)
		: scheduler_{FeldkampScheduler<float>::instance(geo)}
		{
		}

		auto Preloader::process(input_type&& input) -> void
		{
			if(!input.valid())
			{
				sort();
				split();
				distribute();
				finish();
				return;
			}

			input_buf_.emplace_back(std::move(input));
		}

		auto Preloader::sort() -> void
		{
			// align input images on 90°-boundaries
			auto num_proj = input_buf_.size();
			auto step = num_proj / 4;
			for(auto i = 0u; i < step; ++i)
			{
				sorted_input_.emplace_back(std::array<input_type, 4>{
											std::move(input_buf_[i]),
											std::move(input_buf_[i + step]),
											std::move(input_buf_[i + 2 * step]),
											std::move(input_buf_[i + 3 * step])
				});
			}
			input_buf_.clear();
		}

		auto Preloader::split() -> void
		{
			// create subprojections and upload them to their corresponding devices
			auto dev_count = int{};
			ddrf::cuda::check(cudaGetDeviceCount(&dev_count));

			auto chunks = scheduler_.get_chunks();

			for(auto d = 0; d < dev_count; ++d)
			{
					auto chunks_on_dev = chunks[d];
					auto chunk_map = std::map<std::size_t, std::vector<std::array<input_type ,4>>>{};
					for(auto i = 0u; i < chunks_on_dev.size(); ++i)
					{
						auto firstRow = chunks_on_dev[i].first;
						auto lastRow = chunks_on_dev[i].second;
						auto height = lastRow - firstRow + 1;

						auto chunk_vec = std::vector<std::array<input_type, 4>>{};

						for(auto& arr : sorted_input_)
						{
							using data_type = typename input_type::value_type;

							auto splitImg = [height, this](const input_type& src)
							{
								auto width = src.width();
								auto data = ddrf::cuda::make_host_ptr<float>(width, height);

								std::copy(src.data(), src.data() + src.pitch() * height, data.get());
								auto ret = input_type{width, height, std::move(data)};
								return ret;
							};

							chunk_vec.emplace_back(typename decltype(chunk_vec)::value_type {
								splitImg(arr[0]),
								splitImg(arr[1]),
								splitImg(arr[2]),
								splitImg(arr[3])
							});
						}

						chunk_map[i] = std::move(chunk_vec);

					}
					device_chunk_map_[d] = std::move(chunk_map);
			}
		}

		auto Preloader::distribute() -> void
		{
			auto dev_count = int{};
			ddrf::cuda::check(cudaGetDeviceCount(&dev_count));

			for(auto d = 0; d < dev_count; ++d)
				distribution_threads_.emplace_back(&Preloader::uploadAndSend, this, d);
		}

		auto Preloader::uploadAndSend(int device) -> void
		{
			ddrf::cuda::check(cudaSetDevice(device));
			auto chunk_map = device_chunk_map_[device];

			auto uploadArray = [this, device](const std::array<input_type, 4>& arr)
			{
				auto ret = output_type{};

				for(auto i = 0u; i < arr.size(); ++i)
				{
					auto img = arr[i];

					auto width = img.width();
					auto height = img.height();

					auto data = ddrf::cuda::make_device_ptr<float>(width, height);
					ddrf::cuda::copy_async(data, img.container());

					ret[i] = output_img_type(width, height, std::move(data));
					ret[i].setDevice(device);
				}
				return ret;
			};

			for(auto& kv : chunk_map)
			{
				for(auto& arr : kv.second)
				{
					auto dev_arr = uploadArray(arr);
					results_.push(std::move(dev_arr));
				}
			}
		}

		auto Preloader::finish() -> void
		{
			BOOST_LOG_TRIVIAL(debug) << "CUDAPreloader: Received poisonous pill, called finish()";

			for(auto&& t : distribution_threads_)
				t.join();

			results_.push(output_type{});
		}
	}
}
