#ifndef CUDA_FILTER_H_
#define CUDA_FILTER_H_

#include <cstddef>
#include <cstdint>
#include <deque>
#include <future>
#include <map>
#include <memory>
#include <thread>
#include <vector>

#include <ddrf/Image.h>
#include <ddrf/Queue.h>
#include <ddrf/cuda/DeviceMemoryManager.h>
#include <ddrf/cuda/Memory.h>

#include "../common/Geometry.h"

namespace ddafa
{
	namespace cuda
	{
		class Filter
		{
			public:
				using input_type = ddrf::Image<ddrf::cuda::DeviceMemoryManager<float>>;
				using output_type = ddrf::Image<ddrf::cuda::DeviceMemoryManager<float>>;

			public:
				Filter(const ddafa::common::Geometry& geo);
				auto process(input_type&& img) -> void;
				auto wait() -> output_type;

			protected:
				~Filter() = default;

			private:
				auto filterProcessor(int) -> void;
				auto processor(input_type&& img, std::promise<bool> pr) -> void;
				auto finish() -> void;

			private:
				ddrf::Queue<output_type> results_;
				std::vector<std::thread> processor_threads_;
				std::map<int, std::deque<std::future<bool>>> processor_futures_;
				int devices_;
				std::size_t filter_length_;
				std::vector<ddrf::cuda::device_ptr<float, ddrf::cuda::sync_copy_policy>> rs_;
				float tau_;
		};
	}
}


#endif /* CUDAFILTER_H_ */
