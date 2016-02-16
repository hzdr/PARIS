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

namespace ddafa
{
	namespace impl
	{
		struct CUDADeviceDeleter
		{
			auto operator()(void*) -> void;
		};
	}
}

#endif /* CUDADEVICEDELETER_H_ */
