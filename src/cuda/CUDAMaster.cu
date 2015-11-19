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
		: device_{device_num}, number_of_workers_{1}
		{
			cudaDeviceProp properties;
			cudaError_t err = cudaGetDeviceProperties(&properties, device_);

			if(err != cudaSuccess)
				throw std::runtime_error("CUDAMaster: Invalid device #" + std::to_string(device_));

			if(properties.concurrentKernels == 0)
				std::cout << "CUDAMaster: WARNING: Device #" << device_ << " does not support concurrent kernels."
							<< std::endl;
			else
				std::cout << "CUDAMaster: Device #" << device_ << " supports concurrent kernels." << std::endl;

			// this is ridiculous but CUDA doesn't supply us with the number of resident grids per device
			switch(properties.major)
			{
				case 2:
					number_of_workers_ = 16;
					break;

				case 3:
					switch(properties.minor)
					{
						case 0:
							number_of_workers_ = 16;
							break;

						case 2:
							number_of_workers_ = 4;
							break;

						case 5:
						case 7:
							number_of_workers_ = 32;
							break;

						default:
							throw std::runtime_error("CUDAMaster: Unsupported Compute Capability on device #"
														+ std::to_string(device_)
														+ " (Compute Capability is 3."
														+ std::to_string(properties.minor) + ")");
					}
					break;

				case 5:
					switch(properties.minor)
					{
						case 0:
						case 2:
							number_of_workers_ = 32;
							break;

						case 3:
							number_of_workers_ = 16;
							break;

						default:
							throw std::runtime_error("CUDAMaster: Unsupported Compute Capability on device #"
														+ std::to_string(device_)
														+ " (Compute Capability is 5."
														+ std::to_string(properties.minor) + ")");
					}
					break;

				default:
						throw std::runtime_error("CUDAMaster: Unsupported Compute Capability on device #"
														+ std::to_string(device_)
														+ " (Compute Capability is "
														+ std::to_string(properties.major) + "."
														+ std::to_string(properties.minor) + ")");
			}

			std::cout << "CUDAMaster: Device #" << device_ << " supports " << number_of_workers_
					<< " concurrent kernels." << std::endl;

			std::cout << "CUDAMaster for device #" << device_ << " constructed." << std::endl;
		}

		CUDAMaster::CUDAMaster(CUDAMaster&& other)
		: device_{other.device_}, number_of_workers_{other.number_of_workers_}
		{
		}

		CUDAMaster::~CUDAMaster()
		{
			std::cout << "CUDAMaster for device #" << device_ << " destructed." << std::endl;
		}

		void CUDAMaster::start()
		{
			cudaError_t err = cudaSetDevice(device_); // bind device to current thread
			if(err != cudaSuccess)
			{
				switch(err)
				{
					case cudaErrorInvalidDevice:
						throw std::runtime_error("CUDAMaster: Invalid device #" + std::to_string(device_));

					case cudaErrorDeviceAlreadyInUse:
						throw std::runtime_error("CUDAMaster: Device #" + std::to_string(device_)
													+ " already in use");

					default:
						throw std::runtime_error("CUDAMaster: Unknown error while binding device #" +
													std::to_string(device_) + " to current thread.");
				}
			}
		}

		void CUDAMaster::stop()
		{

		}

		int CUDAMaster::workerCount() const noexcept
		{
			return number_of_workers_;
		}

		ddafa::master_worker::Task<CUDAMaster::task_type>
		CUDAMaster::createTask(const ddafa::image::Image* img_ptr)
		{
			std::cout << "CUDAMaster: STUB: createTask() called" << std::endl;
			return ddafa::master_worker::Task<task_type>(0, nullptr, nullptr);
		}
	}
}
