/*
 * CUDAWeighting.cu
 *
 *  Created on: 19.11.2015
 *      Author: Jan Stephan
 *
 *      CUDAWeighting manages the concrete implementation of weighting the projections. Implementation file.
 */

#include <cstdint>

#include "CUDAWeighting.h"

#include "../image/Image.h"

namespace ddafa
{
	namespace impl
	{
		CUDAWeighting::CUDAWeighting()
		{
		}

		CUDAWeighting::~CUDAWeighting()
		{
		}

		void CUDAWeighting::process(CUDAWeighting::input_image_type&& img)
		{
		}

		CUDAWeighting::output_image_type wait()
		{
			return CUDAWeighting::output_image_type();
		}
	}
}
