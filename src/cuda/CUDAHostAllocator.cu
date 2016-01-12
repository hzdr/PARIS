/*
 * CUDAHostAllocator.cu
 *
 *  Created on: 12.01.2016
 *      Author: Jan Stephan
 *
 *      Custom allocator for allocating pinned host memory using CUDA's memory management functions. Implementation file.
 */

#include <cstddef>

#include "CUDAAssert.h"
#include "CUDAHostAllocator.h"

namespace ddafa
{
	namespace impl
	{
		void* CUDAHostAllocator::allocate(std::size_t bytes)
		{
			void *ptr;
			assertCuda(cudaMallocHost(&ptr, bytes));
			return ptr;
		}

		void CUDAHostAllocator::deallocate(void* ptr)
		{
			assertCuda(cudaFreeHost(ptr));
		}
	}
}
