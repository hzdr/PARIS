/*
 * CUDAFeldkamp.cu
 *
 *  Created on: 12.11.2015
 *      Author: Jan Stephan
 *
 *      This class is the concrete backprojection implementation for the Stage class. Implementation file.
 */

#include <ddrf/Image.h>
#include <ddrf/cuda/Check.h>

#include "CUDAFeldkamp.h"

namespace ddafa
{
	namespace impl
	{
		CUDAFeldkamp::CUDAFeldkamp()
		{
			auto device_count = int{};
			ddrf::cuda::check(cudaGetDeviceCount(&device_count));
		}

		auto CUDAFeldkamp::process(CUDAFeldkamp::input_type&& img) -> void
		{
		}

		auto CUDAFeldkamp::wait() -> CUDAFeldkamp::output_type
		{
			return CUDAFeldkamp::output_type();
		}
	}
}
