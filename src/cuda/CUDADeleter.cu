/*
 * CUDADeleter.cu
 *
 *  Created on: 19.11.2015
 *      Author: Jan Stephan
 *
 *      A custom deleter for CUDA memory that is managed by smart pointers. Implementation file.
 */

#include <stdexcept>
#include <string>

#include "CUDAAssert.h"
#include "CUDADeleter.h"

namespace ddafa
{
	namespace impl
	{
		void CUDADeleter::operator()(void *p)
		{
			if(p != nullptr)
				assertCuda(cudaFree(p));
		}
	}
}

