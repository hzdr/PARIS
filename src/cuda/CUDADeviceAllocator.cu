/*
 * CUDADeviceAllocator.cu
 *
 *  Created on: 12.01.2016
 *      Author: Jan Stephan
 *
 *      Custom allocator for allocating GPU memory using CUDA's memory management functions. Implementation file.
 */

#include <cstddef>

#include "CUDAAssert.h"
#include "CUDADeviceAllocator.h"

namespace ddafa
{
	namespace impl
	{
		void* CUDADeviceAllocator::allocate(std::size_t bytes)
		{
			void *ptr;
			assertCuda(cudaMalloc(&ptr, bytes));
			return ptr;
		}

		void CUDADeviceAllocator::deallocate(void* ptr)
		{
			assertCuda(cudaFree(ptr));
		}
	}
}
