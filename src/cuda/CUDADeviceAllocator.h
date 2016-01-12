/*
 * CUDADeviceAllocator.h
 *
 *  Created on: 12.01.2016
 *      Author: Jan Stephan
 *
 *      Custom allocator for allocating GPU memory using CUDA's memory management functions.
 */

#ifndef CUDADEVICEALLOCATOR_H_
#define CUDADEVICEALLOCATOR_H_

#include <cstddef>

namespace ddafa
{
	namespace impl
	{
		class CUDADeviceAllocator
		{
			public:
				void* allocate(std::size_t bytes);
				void deallocate(void* ptr);

			protected:
				~CUDADeviceAllocator() = default;
		};
	}
}

#endif /* CUDADEVICEALLOCATOR_H_ */
