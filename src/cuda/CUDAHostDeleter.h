/*
 * CUDAHostDeleter.h
 *
 *  Created on: 12.01.2016
 *      Author: Jan Stephan
 *
 *      A custom deleter for CUDA host memory that is managed by smart pointers.
 */

#ifndef CUDAHOSTDELETER_H_
#define CUDAHOSTDELETER_H_

#include "CUDAHostAllocator.h"

namespace ddafa
{
	namespace impl
	{
		class CUDAHostDeleter : private CUDAHostAllocator
		{
			void operator()(void *p);
		};
	}
}


#endif /* CUDAHOSTDELETER_H_ */
