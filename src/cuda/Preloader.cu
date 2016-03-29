#include <cstddef>
#include <iterator>
#include <utility>

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
		, processor_thread_{&Preloader::processor, this}
		{
			CHECK(cudaGetDeviceCount(&devices_));
		}

		Preloader::~Preloader()
		{
			processor_thread_.join();
		}

		auto Preloader::process(input_type&& img) -> void
		{
			imgs_.push(std::move(img));
		}

		auto Preloader::processor() -> void
		{
			while(true)
			{
				auto img = imgs_.take();
				if(!img.valid())
				{
					finish();
					break;
				}
				split(std::move(img));
				distribute_first();
			}
		}

		auto Preloader::split(input_type img) -> void
		{
			// create subprojections
			auto chunks = scheduler_.get_chunks();

			for(auto d = 0; d < devices_; ++d)
			{
				auto chunks_on_dev = chunks[d];
				for(auto i = 0u; i < chunks_on_dev.size(); ++i)
				{
					auto firstRow = chunks_on_dev[i].first;
					auto lastRow = chunks_on_dev[i].second;
					auto rows = lastRow - firstRow + 1;

					auto extract = [this](const input_type& src, std::uint32_t first, std::uint32_t height)
					{
						auto width = src.width();
						auto dest = ddrf::cuda::make_host_ptr<float>(width, height);

						// explicit call to cudaMemcpy2D as we are only copying parts of the input data
						CHECK(cudaMemcpy2D(dest.get(), dest.pitch(),
									src.data() + first * width, src.pitch(),
									width * sizeof(float), height, cudaMemcpyHostToHost));

						auto ret = input_type{width, height, src.index(), std::move(dest)};
						return ret;
					};

					remaining_[d][i].emplace_back(extract(img, firstRow, rows));
				}
			}
		}

		auto Preloader::distribute_first() -> void
		{
			auto distribution_threads = std::vector<std::thread>{};
			for(auto d = 0; d < devices_; ++d)
			{
				/* This looks ugly, but is quite simple: For each available CUDA device,
				 * take the first associated input image out of the queue and pass it to
				 * the upload thread. Then, pop the (now empty) first entry in the queue.
				 */
				auto& queue = std::begin(remaining_[d])->second;
				if(queue.empty())
					continue;
				distribution_threads.emplace_back(&Preloader::uploadAndSend, this, d, std::move(queue.front()));
				queue.pop_front();
			}


			for(auto&& t : distribution_threads)
				t.join();
		}

		auto Preloader::uploadAndSend(int device, input_type img) -> void
		{
			BOOST_LOG_TRIVIAL(debug) << "cuda::Preloader: Uploading image #" << img.index() << " to device #" << device;
			CHECK(cudaSetDevice(device));

			auto upload = [this, device](const input_type& input)
			{
				auto ret = output_type{};

				auto width = input.width();
				auto height = input.height();

				auto data = ddrf::cuda::make_device_ptr<float>(width, height);
				ddrf::cuda::copy_sync(data, input.container());

				ret = output_type{width, height, input.index(), std::move(data)};
				ret.setDevice(device);

				return ret;
			};

			results_.push(upload(img));
		}

		auto Preloader::finish() -> void
		{
			BOOST_LOG_TRIVIAL(debug) << "cuda::Preloader: Received poisonous pill, finishing...";

			results_.push(output_type{});
			BOOST_LOG_TRIVIAL(info) << "cuda::Preloader: Done.";
		}

		auto Preloader::wait() -> output_type
		{
			return results_.take();
		}
	}
}
