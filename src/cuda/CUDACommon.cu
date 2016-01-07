/*
 * CUDACommon.cu
 *
 *  Created on: 17.12.2015
 *      Author: Jan Stephan
 */

#include "CUDACommon.h"

namespace ddafa
{
	namespace impl
	{
		__device__ unsigned int getX()
		{
			return blockIdx.x * blockDim.x + threadIdx.x;
		}

		__device__ unsigned int getY()
		{
			return blockIdx.y * blockDim.y + threadIdx.y;
		}
	}
}
