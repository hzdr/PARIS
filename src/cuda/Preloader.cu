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
				split();
				distribute();
				finish();
				return;
			}

			input_buf_.emplace_back(std::move(input));
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
				auto chunk_map = std::map<std::size_t, std::vector<input_type>>{};
				for(auto i = 0u; i < chunks_on_dev.size(); ++i)
				{
					auto firstRow = chunks_on_dev[i].first;
					auto lastRow = chunks_on_dev[i].second;
					auto rows = lastRow - firstRow + 1;

					auto chunk_vec = std::vector<input_type>{};

					auto extract = [this](const input_type& src, std::uint32_t first, std::uint32_t height)
					{
						auto width = src.width();
						auto dest = ddrf::cuda::make_host_ptr<float>(width, height);

						cudaMemcpy2D(dest.get(), dest.pitch(), src.data() + first * width, src.pitch(), width * sizeof(float), height, cudaMemcpyHostToHost);

						auto ret = input_type{width, height, std::move(dest)};
						return ret;
					};

					for(auto& img : input_buf_)
						chunk_vec.emplace_back(extract(img, firstRow, rows));

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
			BOOST_LOG_TRIVIAL(debug) << "cuda::Preloader: Uploading to device #" << device;
			ddrf::cuda::check(cudaSetDevice(device));
			auto chunk_map = device_chunk_map_[device];

			auto upload = [this, device](const input_type& img)
			{
				auto ret = output_type{};

				auto width = img.width();
				auto height = img.height();

				auto data = ddrf::cuda::make_device_ptr<float>(width, height);
				ddrf::cuda::copy_async(data, img.container());

				ret = output_type{width, height, std::move(data)};
				ret.setDevice(device);

				return ret;
			};

			for(auto& kv : chunk_map)
			{
				for(auto& img : kv.second)
				{
					auto dev_img = upload(img);
					results_.push(std::move(dev_img));
				}
			}
		}

		auto Preloader::finish() -> void
		{
			BOOST_LOG_TRIVIAL(debug) << "CUDAPreloader: Received poisonous pill, called finish()";

			for(auto&& t : distribution_threads_)
				t.join();

			results_.push(output_type{});
			input_buf_.clear();
		}

		auto Preloader::wait() -> output_type
		{
			return results_.take();
		}
	}
}
