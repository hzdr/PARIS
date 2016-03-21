#ifndef CUDA_WEIGHTING_H_
#define CUDA_WEIGHTING_H_

#include <cstddef>
#include <cstdint>
#include <deque>
#include <future>
#include <map>
#include <thread>
#include <vector>

#include <ddrf/Queue.h>
#include <ddrf/Image.h>
#include <ddrf/cuda/DeviceMemoryManager.h>
#include <ddrf/cuda/HostMemoryManager.h>
#include <ddrf/default/MemoryManager.h>

#include "../common/Geometry.h"

namespace ddafa
{
	namespace cuda
	{
		class Weighting
		{
			public:
				using input_type = ddrf::Image<ddrf::cuda::DeviceMemoryManager<float>>;
				using output_type = ddrf::Image<ddrf::cuda::DeviceMemoryManager<float>>;

			public:
				Weighting(const ddafa::common::Geometry&);
				auto process(input_type&&) -> void;
				auto wait() -> output_type;

			protected:
				~Weighting() = default;

			private:
				auto processor(input_type&&, std::promise<bool>) -> void;
				auto finish() -> void;

			private:
				ddafa::common::Geometry geo_;
				ddrf::Queue<output_type> results_;
				float h_min_;
				float v_min_;
				float d_dist_;
				int devices_;
				std::vector<std::thread> processor_threads_;
				std::map<int, std::deque<std::future<bool>>> processor_futures_;
		};
	}
}


#endif /* CUDAWEIGHTING_H_ */
