/*
 * CUDAWeighting.h
 *
 *  Created on: 19.11.2015
 *      Author: Jan Stephan
 *
 *      CUDAWeighting manages the concrete implementation of weighting the projections.
 */

#ifndef CUDAWEIGHTING_H_
#define CUDAWEIGHTING_H_

#include <cstddef>
#include <cstdint>
#include <thread>
#include <vector>

#include <ddrf/Queue.h>
#include <ddrf/Image.h>
#include <ddrf/default/Image.h>
#include <ddrf/cuda/Image.h>
#include <ddrf/cuda/HostMemoryManager.h>

#include "../common/Geometry.h"

namespace ddafa
{
	namespace impl
	{
		class CUDAWeighting
		{
			public:
				using input_type = ddrf::Image<ddrf::def::Image<float, ddrf::cuda::HostMemoryManager<float>>>;
				using output_type = ddrf::Image<ddrf::cuda::Image<float>>;

			public:
				CUDAWeighting(const ddafa::common::Geometry&);
				auto process(input_type&&) -> void;
				auto wait() -> output_type;

			protected:
				~CUDAWeighting() = default;

			private:
				auto processor(const input_type&, int) -> void;
				auto finish() -> void;

			private:
				ddafa::common::Geometry geo_;
				ddrf::Queue<output_type> results_;
				float h_min_;
				float v_min_;
				float d_dist_;
				int devices_;
				std::vector<std::thread> processor_threads_;
		};
	}
}


#endif /* CUDAWEIGHTING_H_ */
