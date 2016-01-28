/*
 * CUDAHostDeleter.cu
 *
 *  Created on: 12.01.2016
 *      Author: Jan Stephan
 *
 *      A custom deleter for CUDA host memory that is managed by smart pointers. Implementation file.
 */

#include "CUDAAssert.h"
#include "CUDAHostDeleter.h"

namespace ddafa
{
	namespace impl
	{
		void CUDAHostDeleter::operator()(void *p)
		{
			assertCuda(cudaFreeHost(p));
		}
	}
}
