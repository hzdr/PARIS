/*
 * CUDAMaster.cu
 *
 *  Created on: 12.11.2015
 *      Author: Jan Stephan
 *
 *      CUDA implementation policy for the Master class. Implementation file.
 */

#include <iostream>
#include <stdexcept>
#include <string>

#include "CUDAMaster.h"

namespace ddafa
{
	namespace impl
	{
		CUDAMaster::CUDAMaster(int device_num)
		: device_{device_num}
		{
			cudaDeviceProp properties;
			cudaError_t err = cudaGetDeviceProperties(&properties, device_);

			if(err != cudaSuccess)
				throw std::runtime_error("CUDAMaster: Invalid device #" + std::to_string(device_));

			std::cout << "CUDAMaster for device #" << device_ << " constructed." << std::endl;
		}

		CUDAMaster::CUDAMaster(CUDAMaster&& other)
		: device_{other.device_}
		{
		}

		CUDAMaster::~CUDAMaster()
		{
			std::cout << "CUDAMaster for device #" << device_ << " destructed." << std::endl;
		}
	}
}
