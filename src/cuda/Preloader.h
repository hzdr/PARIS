#ifndef CUDA_PRELOADER_H_
#define CUDA_PRELOADER_H_

#include <cstddef>
#include <deque>
#include <map>
#include <thread>

#include <ddrf/Image.h>
#include <ddrf/Queue.h>
#include <ddrf/cuda/DeviceMemoryManager.h>
#include <ddrf/cuda/HostMemoryManager.h>
#include <ddrf/default/MemoryManager.h>

#include "../common/Geometry.h"

#include "FeldkampScheduler.h"

namespace ddafa
{
	namespace cuda
	{
		class Preloader
		{
			public:
				using input_type = ddrf::Image<ddrf::cuda::HostMemoryManager<float>>;
				using output_type = ddrf::Image<ddrf::cuda::DeviceMemoryManager<float>>;

			public:
				Preloader(const common::Geometry& geo);
				auto process(input_type&& input) -> void;
				auto wait() -> output_type;

			protected:
				~Preloader();

			private:
				auto processor() -> void;
				auto split(input_type) -> void;
				auto distribute_first() -> void;
				auto finish() -> void;

				auto uploadAndSend(int, input_type) -> void;

			private:
				ddrf::Queue<input_type> imgs_;
				ddrf::Queue<output_type> results_;
				int devices_;

				std::thread processor_thread_;

				FeldkampScheduler<float> scheduler_;

				std::map<int, std::map<std::size_t, std::deque<input_type>>> remaining_;
		};
	}
}

#endif /* CUDA_PRELOADER_H_ */
