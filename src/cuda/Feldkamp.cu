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

#include "../common/Geometry.h"
#include "Feldkamp.h"
#include "FeldkampScheduler.h"

namespace ddafa
{
	namespace cuda
	{
		Feldkamp::Feldkamp(const common::Geometry& geo)
		: scheduler_{FeldkampScheduler<float>::instance(geo)}
		{
			auto device_count = int{};
			ddrf::cuda::check(cudaGetDeviceCount(&device_count));
		}

		auto Feldkamp::process(input_type&& img) -> void
		{
		}

		auto Feldkamp::wait() -> output_type
		{
			return output_type{};
		}
	}
}
