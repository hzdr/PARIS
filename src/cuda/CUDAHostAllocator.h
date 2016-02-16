/*
 * CUDAHostAllocator.h
 *
 *  Created on: 12.01.2016
 *      Author: Jan Stephan
 *
 *      Custom allocator for allocating pinned host memory using CUDA's memory management functions.
 */

#ifndef CUDAHOSTALLOCATOR_H_
#define CUDAHOSTALLOCATOR_H_

#include <cstddef>

#ifndef __CUDACC__
#include <cuda_runtime.h>
#endif

#include "CUDAAssert.h"

namespace ddafa
{
	namespace impl
	{
		template <typename T>
		class CUDAHostAllocator
		{
			public:
				using value_type = T;
				using pointer = T*;
				using const_pointer = const T*;
				using reference = T&;
				using const_reference = const T&;
				using size_type = std::size_t;
				using difference_type = std::ptrdiff_t;

			public:
				auto allocate(size_type width, size_type height, size_type* pitch) -> pointer
				{
					auto p = static_cast<void*>(nullptr);
					assertCuda(cudaMallocHost(&p, width * height * sizeof(value_type)));
					*pitch = width * sizeof(value_type);
					return static_cast<pointer>(p);
				}

				auto deallocate(pointer p, size_type) -> void
				{
					assertCuda(cudaFreeHost(p));
				}

			protected:
				~CUDAHostAllocator() = default;
		};
	}
}


#endif /* CUDAHOSTALLOCATOR_H_ */
