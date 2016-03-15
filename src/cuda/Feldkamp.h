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

#include "../common/Geometry.h"

#include "FeldkampScheduler.h"

namespace ddafa
{
	namespace cuda
	{
		class Feldkamp
		{
			private:
				using input_type = ddrf::Image<ddrf::cuda::Image<float>>;
				using output_type = ddrf::Image<ddrf::def::Image<float, ddrf::cuda::HostMemoryManager<float>>>;

			public:
				Feldkamp(const common::Geometry&);
				auto process(input_type&&) -> void;
				auto wait() -> output_type;

			protected:
				~Feldkamp() = default;

			private:
				FeldkampScheduler<float> scheduler_;
		};
	}
}


#endif /* CUDAFELDKAMP_H_ */
