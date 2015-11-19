/*
 * CUDAFeldkamp.cu
 *
 *  Created on: 12.11.2015
 *      Author: Jan Stephan
 *
 *      This class is the concrete backprojection implementation for the Stage class. Implementation file.
 */
#
#include <stdexcept>

#include "../image/Image.h"
#include "../master_worker/Master.h"

#include "CUDAFeldkamp.h"

namespace ddafa
{
	namespace impl
	{
		CUDAFeldkamp::CUDAFeldkamp()
		{
			int device_count;
			cudaError_t err = cudaGetDeviceCount(&device_count);

			switch(err)
			{
				case cudaSuccess:
					break;

				case cudaErrorNoDevice:
					throw std::runtime_error("CUDAFeldkamp: No CUDA devices found.");

				case cudaErrorInsufficientDriver:
					throw std::runtime_error("CUDAFeldkamp: Insufficient driver.");
			}

			for(int i = 0; i < device_count; ++i)
				masters_.emplace_back(i);

			for(auto&& master : masters_)
				master_threads_.emplace_back(&ddafa::master_worker::Master<CUDAMaster, int&>::start, &master);
		}

		CUDAFeldkamp::~CUDAFeldkamp()
		{
		}

		void CUDAFeldkamp::process(ddafa::image::Image&& img)
		{
			// do NOT delete this pointer
			ddafa::image::Image* img_ptr = &img;
			for(auto&& master : masters_)
				master.input(img_ptr);
		}

		ddafa::image::Image CUDAFeldkamp::wait()
		{
			for(auto&& thread : master_threads_)
				thread.join();
			return ddafa::image::Image();
		}
	}
}
