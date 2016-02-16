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
		__device__ auto getX() -> unsigned int
		{
			return blockIdx.x * blockDim.x + threadIdx.x;
		}

		__device__ auto getY() -> unsigned int
		{
			return blockIdx.y * blockDim.y + threadIdx.y;
		}
	}
}
