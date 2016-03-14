/*
 * CUDAFeldkamp.h
 *
 *  Created on: 12.11.2015
 *      Author: Jan Stephan
 *
 *      This class is the concrete backprojection implementation for the Stage class.
 */

#ifndef CUDAFELDKAMP_H_
#define CUDAFELDKAMP_H_

#include <ddrf/Image.h>
#include <ddrf/cuda/HostMemoryManager.h>
#include <ddrf/cuda/Image.h>
#include <ddrf/cuda/Memory.h>
#include <ddrf/default/Image.h>

namespace ddafa
{
	namespace impl
	{
		class CUDAFeldkamp
		{
			private:
				using input_type = ddrf::Image<ddrf::cuda::Image<float>>;
				using output_type = ddrf::Image<ddrf::def::Image<float, ddrf::cuda::HostMemoryManager<float>>>;

			public:
				CUDAFeldkamp();
				auto process(input_type&&) -> void;
				auto wait() -> output_type;

			protected:
				~CUDAFeldkamp() = default;
		};
	}
}


#endif /* CUDAFELDKAMP_H_ */
