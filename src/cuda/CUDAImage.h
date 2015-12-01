/*
 * CUDAImage.h
 *
 *  Created on: 19.11.2015
 *      Author: Jan Stephan
 *
 *      CUDAImage provides an implementation for the Image class that loads the image into GPU memory.
 */

#ifndef CUDAIMAGE_H_
#define CUDAIMAGE_H_

#include <cstddef>
#include <memory>
#include <stdexcept>

#ifndef __CUDACC__
#include <cuda_runtime.h>
#endif

#include "CUDADeleter.h"

namespace ddafa
{
	namespace impl
	{
		template <typename Data>
		class CUDAImage
		{
			public:
				using deleter_type = CUDADeleter;

			public:
				std::unique_ptr<Data, deleter_type> allocate(std::size_t size)
				{
					void *ptr;
					cudaError_t err = cudaMalloc(&ptr, size);
					switch(err)
					{
						case cudaErrorMemoryAllocation:
							throw std::runtime_error("CUDAImage: Error while allocating memory");

						case cudaSuccess:
						default:
							break;
					}

					return std::unique_ptr<Data, deleter_type>(static_cast<Data*>(ptr));
				}

				void copy(const Data* src, Data* dest, std::size_t size)
				{
					cudaError_t err = cudaMemcpy(dest, src, size, cudaMemcpyDeviceToDevice);
					switch(err)
					{
						case cudaErrorInvalidValue:
							throw std::runtime_error("CUDAImage: Invalid value");

						case cudaErrorInvalidDevicePointer:
							throw std::runtime_error("CUDAImage: Invalid device value");

						case cudaErrorInvalidMemcpyDirection:
							throw std::runtime_error("CUDAImage: Invalid memcpy direction");

						case cudaSuccess:
						default:
							break;
					}
				}

				void setDevice(int device_id)
				{
					device_ = device_id;
				}

				int device()
				{
					return device_;
				}

			private:
				int device_;
		};
	}
}


#endif /* CUDAIMAGE_H_ */
