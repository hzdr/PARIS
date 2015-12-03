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

#include "CUDADeleter.h"

namespace ddafa
{
	namespace impl
	{
		void CUDADeleter::operator()(void *p)
		{
			cudaError_t err = cudaFree(p);
			if(err != cudaSuccess)
				throw std::runtime_error("CUDADeleter::operator(): " + std::string(cudaGetErrorString(err)));
		}
	}
}

