#include <algorithm>
#include <cstddef>
#include <future>
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
			ddrf::cuda::check(cudaGetDeviceCount(&devices_));
			auto pr = std::promise<bool>{};
			processor_futures_.emplace_back(pr.get_future());
			pr.set_value(true);
		}

		auto Preloader::process(input_type&& img) -> void
		{
			if(!img.valid())
			{
				finish();
				return;
			}

			auto pr = std::promise<bool>{};
			processor_futures_.emplace_back(pr.get_future());
			processor_threads_.emplace_back(&Preloader::processor, this, std::move(img), std::move(pr));
		}

		auto Preloader::processor(input_type&& img, std::promise<bool> pr) -> void
		{
			auto future = std::move(processor_futures_.front());
			processor_futures_.pop_front();
			auto start = future.get();
			start = !start;

			distribute(split(std::move(img)));

			pr.set_value(true);
		}

		auto Preloader::split(input_type&& img) -> std::map<int, std::map<std::size_t, input_type>>
		{
			// create subprojections and upload them to their corresponding devices
			auto chunks = scheduler_.get_chunks();

			auto device_chunk_map = std::map<int, std::map<std::size_t, input_type>>{};
			for(auto d = 0; d < devices_; ++d)
			{
				auto chunks_on_dev = chunks[d];
				auto chunk_map = std::map<std::size_t, input_type>{};
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

						// explicit call to cudaMemcpy2D as we are only copying parts of the input data
						cudaMemcpy2D(dest.get(), dest.pitch(),
									src.data() + first * width, src.pitch(),
									width * sizeof(float), height, cudaMemcpyHostToHost);

						auto ret = input_type{width, height, std::move(dest)};
						return ret;
					};

					chunk_map[i] = extract(img, firstRow, rows);
				}
				device_chunk_map[d] = std::move(chunk_map);
			}

			return device_chunk_map;
		}

		auto Preloader::distribute(std::map<int, std::map<std::size_t, input_type>> map) -> void
		{
			auto distribution_threads = std::vector<std::thread>{};
			for(auto d = 0; d < devices_; ++d)
				distribution_threads.emplace_back(&Preloader::uploadAndSend, this, d, std::move(map[d]));

			for(auto&& t : distribution_threads)
				t.join();
		}

		auto Preloader::uploadAndSend(int device, std::map<std::size_t, input_type> map) -> void
		{
			BOOST_LOG_TRIVIAL(debug) << "cuda::Preloader: Uploading to device #" << device;
			ddrf::cuda::check(cudaSetDevice(device));

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

			for(auto& ii : map)
			{
				results_.push(upload(ii.second));
			}
		}

		auto Preloader::finish() -> void
		{
			BOOST_LOG_TRIVIAL(debug) << "CUDAPreloader: Received poisonous pill, called finish()";

			for(auto&& t : processor_threads_)
				t.join();

			for(auto& f: processor_futures_)
			{
				auto val = f.get();
				val = !val;
			}

			results_.push(output_type{});
		}

		auto Preloader::wait() -> output_type
		{
			return results_.take();
		}
	}
}
