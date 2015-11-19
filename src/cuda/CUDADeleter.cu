/*
 * CUDADeleter.cu
 *
 *  Created on: 19.11.2015
 *      Author: Jan Stephan
 *
 *      A custom deleter for CUDA memory that is managed by smart pointers. Implementation file.
 */

#include <stdexcept>

#include "CUDADeleter.h"

namespace ddafa
{
	namespace impl
	{
		void CUDADeleter::operator()(void *p)
		{
			cudaError_t err = cudaFree(p);
			switch(err)
			{
				case cudaErrorInvalidDevicePointer:
					throw std::runtime_error("CUDADeleter: Invalid device pointer");

				case cudaErrorInitializationError:
					throw std::runtime_error("CUDADeleter: Initialization error");

				case cudaSuccess:
				default:
					break;
			}
		}
	}
}

