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
#include <string>

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
					if(err != cudaSuccess)
						throw std::runtime_error("CUDAImage::allocate: " + std::string(cudaGetErrorString(err)));

					return std::unique_ptr<Data, deleter_type>(static_cast<Data*>(ptr));
				}

				void copy(const Data* src, Data* dest, std::size_t size)
				{
					cudaError_t err = cudaMemcpy(dest, src, size, cudaMemcpyDeviceToDevice);
					if(err != cudaSuccess)
						throw std::runtime_error("CUDAImage::copy: " + std::string(cudaGetErrorString(err)));
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
