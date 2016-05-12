#include <cstddef>
#include <future>
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
					distribute_rest();
					finish();
					break;
				}
				split(std::move(img));
				distribute_first();
			}
		}

		auto Preloader::split(input_type img) -> void
		{
			for(auto d = 0; d < devices_; ++d)
			{
				auto subproj_num = scheduler_.get_subproj_num(d);
				for(auto i = 0u; i < subproj_num; ++i)
				{
					auto subproj_dims = scheduler_.get_subproj_dims(d, i);
					auto firstRow = subproj_dims.first;
					auto lastRow = subproj_dims.second;
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

		auto Preloader::distribute_rest() -> void
		{
			BOOST_LOG_TRIVIAL(debug) << "cuda::Preloader: Uploading remaining subprojections.";

			auto futures = std::vector<std::future<void>>{};

			for(auto d = 0; d < devices_; ++d)
			{
				auto map = remaining_.at(d);
				futures.emplace_back(std::async(std::launch::async, [&map, d, this]()
						{
							for(auto& p : map)
							{
								for(auto& img : p.second)
									uploadAndSend(d, std::move(img));
							}
						}));
			}

			for(auto& f : futures)
				f.wait();
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
