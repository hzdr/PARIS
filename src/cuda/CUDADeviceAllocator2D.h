/*
 * CUDADeviceAllocator2D.h
 *
 *  Created on: 02.02.2016
 *      Author: Jan Stephan
 */

#ifndef CUDADEVICEALLOCATOR2D_H_
#define CUDADEVICEALLOCATOR2D_H_

namespace ddafa
{
	namespace impl
	{
		template <typename T>
		class CUDADeviceAllocator2D
		{
			public:
				using value_type = T;
				using pointer = T*;
				using const_pointer = const T*;
				using reference = T&;
				using const_reference = const T&;
				using size_type = std::size_t;
				using difference_type = std::ptrdiff_t;

			public:
				pointer allocate(size_type width, size_type height, size_type* pitch)
				{
					void *p;
					assertCuda(cudaMallocPitch(&p, pitch, width * sizeof(value_type), height));
					return static_cast<pointer>(p);
				}

				void deallocate(pointer p, size_type)
				{
					assertCuda(cudaFree(p));
				}

			protected:
				~CUDADeviceAllocator2D() = default;
		};
	}
}



#endif /* CUDADEVICEALLOCATOR2D_H_ */
