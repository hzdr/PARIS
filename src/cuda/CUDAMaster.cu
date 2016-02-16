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

#define BOOST_ALL_DYN_LINK
#include <boost/log/trivial.hpp>

#include "CUDAAssert.h"
#include "CUDAMaster.h"

namespace ddafa
{
	namespace impl
	{
		CUDAMaster::CUDAMaster(int device_num)
		: device_{device_num}, number_of_workers_{1}
		{
			auto properties = cudaDeviceProp{};
			assertCuda(cudaGetDeviceProperties(&properties, device_));

			if(properties.concurrentKernels == 0)
				BOOST_LOG_TRIVIAL(warning) << "CUDAMaster: Device #" << device_ << " does not support concurrent kernels.";
			else
				BOOST_LOG_TRIVIAL(debug) << "CUDAMaster: Device #" << device_ << " supports concurrent kernels.";

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

			BOOST_LOG_TRIVIAL(debug) << "CUDAMaster: Device #" << device_ << " supports " << number_of_workers_
					<< " concurrent kernels.";

			BOOST_LOG_TRIVIAL(debug) << "CUDAMaster for device #" << device_ << " constructed.";
		}

		CUDAMaster::CUDAMaster(CUDAMaster&& other)
		: device_{other.device_}, number_of_workers_{other.number_of_workers_}
		{
		}

		CUDAMaster::~CUDAMaster()
		{
			BOOST_LOG_TRIVIAL(debug) << "CUDAMaster for device #" << device_ << " destructed.";
		}

		auto CUDAMaster::start() -> void
		{
			assertCuda(cudaSetDevice(device_)); // bind device to current thread
		}

		auto CUDAMaster::stop() -> void
		{
		}

		auto CUDAMaster::workerCount() const noexcept -> int
		{
			return number_of_workers_;
		}

		auto CUDAMaster::createTask(const CUDAMaster::image_type* img_ptr) -> CUDAMaster::task_type
		{
			BOOST_LOG_TRIVIAL(warning) << "CUDAMaster: STUB: createTask() called";
			return task_type(0, nullptr, nullptr);
		}
	}
}
