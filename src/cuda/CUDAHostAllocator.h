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

namespace ddafa
{
	namespace impl
	{
		class CUDAHostAllocator
		{
			public:
				void* allocate(std::size_t bytes);
				void deallocate(void* ptr);

			protected:
				~CUDAHostAllocator() = default;
		};
	}
}


#endif /* CUDAHOSTALLOCATOR_H_ */
