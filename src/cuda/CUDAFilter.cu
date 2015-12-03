/*
 * CUDAFilter.cu
 *
 *  Created on: 03.12.2015
 *      Author: Jan Stephan
 *
 *      CUDAFilter takes a weighted projection and applies a filter to it. Implementation file.
 */

#ifdef DDAFA_DEBUG
#include <iostream>
#endif

#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "CUDAFilter.h"

namespace ddafa
{
	namespace impl
	{
		CUDAFilter::CUDAFilter()
		{
			cudaError_t err = cudaGetDeviceCount(&devices_);
			if(err != cudaSuccess)
				throw std::runtime_error("CUDAFilter::CUDAFilter: " + std::string(cudaGetErrorString(err)));
		}

		CUDAFilter::~CUDAFilter()
		{
		}

		void CUDAFilter::process(CUDAFilter::input_type&& img)
		{
			if(!img.valid())
			{
				// received poisonous pill, time to die
				finish();
				return;
			}

			for(int i = 0; i < devices_; ++i)
			{
				cudaSetDevice(i);
				if(img.device() == i)
					processor_threads_.emplace_back(&CUDAFilter::processor, this, std::move(img));
			}
		}

		void CUDAFilter::processor(CUDAFilter::input_type&& img)
		{

		}

		void CUDAFilter::finish()
		{
#ifdef DDAFA_DEBUG
				std::cout << "CUDAFilter: Received poisonous pill, called finish()" << std::endl;
#endif

				for(auto&& t : processor_threads_)
					t.join();

				results_.push(output_type());
		}
	}
}
