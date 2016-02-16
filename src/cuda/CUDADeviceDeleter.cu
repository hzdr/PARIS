/*
 * CUDADeleter.cu
 *
 *  Created on: 19.11.2015
 *      Author: Jan Stephan
 *
 *      A custom deleter for CUDA device memory that is managed by smart pointers. Implementation file.
 */

#include <stdexcept>
#include <string>

#include "CUDAAssert.h"
#include "CUDADeviceDeleter.h"

namespace ddafa
{
	namespace impl
	{
		auto CUDADeviceDeleter::operator()(void* p) -> void
		{
				assertCuda(cudaFree(p));
		}
	}
}
