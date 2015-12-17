/*
 * CUDAFeldkamp.cu
 *
 *  Created on: 12.11.2015
 *      Author: Jan Stephan
 *
 *      This class is the concrete backprojection implementation for the Stage class. Implementation file.
 */

#include <stdexcept>
#include <string>

#include "../image/Image.h"
#include "../master_worker/Master.h"

#include "CUDAAssert.h"
#include "CUDAFeldkamp.h"

namespace ddafa
{
	namespace impl
	{
		CUDAFeldkamp::CUDAFeldkamp()
		{
			int device_count;
			assertCuda(cudaGetDeviceCount(&device_count));

			for(int i = 0; i < device_count; ++i)
				masters_.emplace_back(i);

			for(auto&& master : masters_)
				master_threads_.emplace_back(&master_type::start, &master);
		}

		CUDAFeldkamp::~CUDAFeldkamp()
		{
		}

		void CUDAFeldkamp::process(CUDAFeldkamp::input_type&& img)
		{
			// do NOT delete this pointer
			input_type* img_ptr = &img;
			for(auto&& master : masters_)
				master.input(img_ptr);
		}

		CUDAFeldkamp::output_type CUDAFeldkamp::wait()
		{
			for(auto&& thread : master_threads_)
				thread.join();
			return CUDAFeldkamp::output_type();
		}
	}
}
