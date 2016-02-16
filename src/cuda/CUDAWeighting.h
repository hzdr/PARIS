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

#include "../common/Geometry.h"
#include "../common/Queue.h"
#include "../image/Image.h"
#include "../image/StdImage.h"

#include "CUDAHostAllocator.h"
#include "CUDAHostDeleter.h"
#include "CUDAImage.h"

namespace ddafa
{
	namespace impl
	{
		class CUDAWeighting
		{
			public:
				using input_type = ddafa::image::Image<float, StdImage<float, CUDAHostAllocator<float>, CUDAHostDeleter>>;
				using output_type = ddafa::image::Image<float, CUDAImage<float>>;

			public:
				CUDAWeighting(const ddafa::common::Geometry&);
				auto process(input_type&&) -> void;
				auto wait() -> output_type;

			protected:
				~CUDAWeighting() = default;

			private:
				auto processor(const input_type&, int) -> void;
				auto finish() -> void;
				auto copyToDevice(const input_type&) -> output_type;

			private:
				ddafa::common::Geometry geo_;
				ddafa::common::Queue<output_type> results_;
				float h_min_;
				float v_min_;
				float d_dist_;
				int devices_;
				std::vector<std::thread> processor_threads_;
		};
	}
}


#endif /* CUDAWEIGHTING_H_ */
