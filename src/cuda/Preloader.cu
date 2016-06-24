#include <cstddef>
#include <future>
#include <ios>
#include <iterator>
#include <stdexcept>
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
		: geo_(geo), processor_thread_{&Preloader::processor, this}
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
				auto ex_img = expand(std::move(img));
				split(std::move(ex_img));
				distribute_first();
			}
		}

        auto Preloader::expand(input_type img) -> input_type
        {
            auto& scheduler = FeldkampScheduler::instance(geo_, cuda::volume_type::single_float);
            auto add_top = scheduler.get_additional_proj_lines_top();
            auto add_bot = scheduler.get_additional_proj_lines_bot();

            auto width = img.width();
            auto old_height = img.height();
            auto new_height = old_height + add_top + add_bot;
            auto offset = add_top * width;

            auto ptr = ddrf::cuda::make_host_ptr<float>(width, new_height);
            ddrf::cuda::memset(ptr, 0.f);
            CHECK(cudaMemcpy2D(ptr.get() + offset, ptr.pitch(), img.data(), img.pitch(),
                    width * sizeof(float), old_height, cudaMemcpyHostToHost));

            return input_type{width, new_height, img.index(), std::move(ptr)};
        }

        auto Preloader::split(input_type img) -> void
        {
            auto& scheduler = FeldkampScheduler::instance(geo_, cuda::volume_type::single_float);
            for(auto d = 0; d < devices_; ++d)
            {
                auto subproj_num = scheduler.get_subproj_num(d);
                for(auto i = 0u; i < subproj_num; ++i)
                {
                    auto subproj_dims = scheduler.get_subproj_dims(d, i);
                    auto firstRow = subproj_dims.first;
                    auto lastRow = subproj_dims.second;
                    auto rows = lastRow - firstRow + 1;

                    auto extract = [this](const input_type& src, std::uint32_t first, std::uint32_t rows)
                    {
                        auto width = src.width();

                        auto dest = ddrf::cuda::make_host_ptr<float>(width, rows);

                        // explicit call to cudaMemcpy2D as we are only copying parts of the input data
                        CHECK(cudaMemcpy2D(dest.get(), dest.pitch(),
                                    src.data() + first * width, src.pitch(),
                                    width * sizeof(float), rows, cudaMemcpyHostToHost));

                        auto ret = input_type{width, rows, src.index(), std::move(dest)};
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
				try
				{
					auto& queue = std::begin(remaining_.at(d))->second;
					if(queue.empty())
						continue;
					distribution_threads.emplace_back(&Preloader::uploadAndSend, this, d, std::move(queue.front()));
					queue.pop_front();
				}
				catch(const std::out_of_range&)
				{
					continue;
				}
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
				auto& map = remaining_.at(d);
				if(map.empty())
					continue;
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
			auto& scheduler = FeldkampScheduler::instance(geo_, cuda::volume_type::single_float);
			scheduler.acquire_projection(device);
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
