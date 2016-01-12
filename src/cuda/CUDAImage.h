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

#include <climits>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>

#include "CUDAAssert.h"
#include "CUDADeviceDeleter.h"

namespace ddafa
{
	namespace impl
	{
		template <typename Data, class Deleter = CUDADeviceDeleter>
		class CUDAImage
		{
			public:
				using deleter_type = Deleter;

			public:
				CUDAImage()
				: device_{INT_MIN}
				{
				}

				CUDAImage(const CUDAImage& other)
				: device_{other.device_}
				{
				}

				CUDAImage& operator=(const CUDAImage& rhs)
				{
					device_ = rhs.device_;
					return *this;
				}

				CUDAImage(CUDAImage&& other)
				: device_{other.device_}
				{
				}

				CUDAImage& operator=(CUDAImage&& rhs)
				{
					device_ = rhs.device_;
					return *this;
				}

				std::unique_ptr<Data, deleter_type> allocate(std::size_t size)
				{
					void *ptr;
					assertCuda(cudaMalloc(&ptr, size));
					return std::unique_ptr<Data, deleter_type>(static_cast<Data*>(ptr));
				}

				void copy(const Data* src, Data* dest, std::size_t size)
				{
					assertCuda(cudaMemcpy(dest, src, size, cudaMemcpyDeviceToDevice));
				}

				void setDevice(int device_id)
				{
					device_ = device_id;
				}

				int device()
				{
					return device_;
				}

			protected:
				~CUDAImage()
				{
				}

			private:
				int device_;
		};
	}
}


#endif /* CUDAIMAGE_H_ */
