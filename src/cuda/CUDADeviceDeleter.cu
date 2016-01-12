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

#include "CUDADeviceDeleter.h"

namespace ddafa
{
	namespace impl
	{
		void CUDADeviceDeleter::operator()(void *p)
		{
				deallocate(p);
		}
	}
}

