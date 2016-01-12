/*
 * CUDADeviceDeleter.h
 *
 *  Created on: 19.11.2015
 *      Author: Jan Stephan
 *
 *      A custom deleter for CUDA device memory that is managed by smart pointers.
 */

#ifndef CUDADEVICEDELETER_H_
#define CUDADEVICEDELETER_H_

#include "CUDADeviceAllocator.h"

namespace ddafa
{
	namespace impl
	{
		struct CUDADeviceDeleter : private CUDADeviceAllocator
		{
			void operator()(void *p);
		};
	}
}

#endif /* CUDADEVICEDELETER_H_ */
