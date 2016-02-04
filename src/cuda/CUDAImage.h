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
#include "CUDADeviceAllocator2D.h"
#include "CUDADeviceDeleter.h"

namespace ddafa
{
	namespace impl
	{
		template <typename Data, class Allocator = CUDADeviceAllocator2D<Data>, class Deleter = CUDADeviceDeleter>
		class CUDAImage : public Allocator
		{
			public:
				using deleter_type = Deleter;
				using allocator_type = Allocator;
				using size_type = std::size_t;
				using value_type = Data;

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

				std::unique_ptr<value_type, deleter_type> allocate(size_type width, size_type height, size_type* pitch)
				{
					return std::unique_ptr<value_type, deleter_type>(Allocator::allocate(width, height, pitch));
				}

				void copy(const value_type* src, value_type* dest, size_type width, size_type height, size_type pitch)
				{
					assertCuda(cudaMemcpy2D(dest, pitch,
											src, pitch,
											width, height,
											cudaMemcpyDeviceToDevice));
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
