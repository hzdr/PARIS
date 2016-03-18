#ifndef CUDA_PRELOADER_H_
#define CUDA_PRELOADER_H_

#include <cstddef>
#include <deque>
#include <future>
#include <map>
#include <thread>
#include <vector>

#include <ddrf/Image.h>
#include <ddrf/Queue.h>
#include <ddrf/cuda/Image.h>
#include <ddrf/cuda/HostMemoryManager.h>
#include <ddrf/default/Image.h>

#include "../common/Geometry.h"

#include "FeldkampScheduler.h"

namespace ddafa
{
	namespace cuda
	{
		class Preloader
		{
			public:
				using input_type = ddrf::Image<ddrf::def::Image<float, ddrf::cuda::HostMemoryManager<float>>>;
				using output_type = ddrf::Image<ddrf::cuda::Image<float>>;

			public:
				Preloader(const common::Geometry& geo);
				auto process(input_type&& input) -> void;
				auto wait() -> output_type;

			protected:
				~Preloader() = default;

			private:
				auto processor(input_type&&, std::promise<bool>) -> void;
				auto split(input_type&&) -> std::map<int, std::map<std::size_t, input_type>>;
				auto distribute(std::map<int, std::map<std::size_t, input_type>>) -> void;
				auto finish() -> void;

				auto uploadAndSend(int device, std::map<std::size_t, input_type>) -> void;

			private:
				ddrf::Queue<output_type> results_;
				int devices_;
				std::vector<std::thread> processor_threads_;
				std::deque<std::future<bool>> processor_futures_;
				FeldkampScheduler<float> scheduler_;
		};
	}
}

#endif /* CUDA_PRELOADER_H_ */
