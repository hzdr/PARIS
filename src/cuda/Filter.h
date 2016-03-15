#ifndef CUDA_FILTER_H_
#define CUDA_FILTER_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

#include <ddrf/Image.h>
#include <ddrf/Queue.h>
#include <ddrf/cuda/Image.h>
#include <ddrf/cuda/Memory.h>

#include "../common/Geometry.h"

namespace ddafa
{
	namespace cuda
	{
		class Filter
		{
			public:
				using input_type = ddrf::Image<ddrf::cuda::Image<float>>;
				using output_type = ddrf::Image<ddrf::cuda::Image<float>>;

			public:
				Filter(const ddafa::common::Geometry& geo);
				auto process(input_type&& img) -> void;
				auto wait() -> output_type;

			protected:
				~Filter() = default;

			private:
				auto filterProcessor(int) -> void;
				auto processor(input_type&& img, int device) -> void;
				auto finish() -> void;

			private:
				ddrf::Queue<output_type> results_;
				std::vector<std::thread> processor_threads_;
				int devices_;
				std::size_t filter_length_;
				std::vector<ddrf::cuda::device_ptr<float, ddrf::cuda::sync_copy_policy>> rs_;
				float tau_;
		};
	}
}


#endif /* CUDAFILTER_H_ */
