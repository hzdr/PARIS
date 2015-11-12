/*
 * CUDAFeldkamp.cu
 *
 *  Created on: 12.11.2015
 *      Author: Jan Stephan
 *
 *      This class is the concrete backprojection implementation for the Stage class. Implementation file.
 */

#include <stdexcept>

#include "../master_worker/Master.h"

#include "CUDAFeldkamp.h"

namespace ddafa
{
	namespace impl
	{
		CUDAFeldkamp::CUDAFeldkamp()
		: masters_{}
		{
			int device_count;
			cudaError_t err = cudaGetDeviceCount(&device_count);

			if(err == cudaErrorNoDevice)
				throw std::runtime_error("CUDAFeldkamp: No CUDA devices found.");

			if(err == cudaErrorInsufficientDriver)
				throw std::runtime_error("CUDAFeldkamp: Insufficient driver.");

			for(int i = 0; i < device_count; ++i)
			{
				masters_.emplace_back(ddafa::master_worker::Master<CUDAMaster, int&>(i));
			}

		}
	}
}
