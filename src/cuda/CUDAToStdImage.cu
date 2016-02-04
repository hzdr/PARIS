/*
 * CUDAToStdImage.cu
 *
 *  Created on: 01.12.2015
 *      Author: Jan Stephan
 *
 *      Converts CUDAImage to StdImage. Implementation file.
 */

#include <cstddef>
#ifdef DDAFA_DEBUG
#include <iostream>
#endif
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

#include "CUDAAssert.h"
#include "CUDAToStdImage.h"

namespace ddafa
{
	namespace impl
	{
		CUDAToStdImage::CUDAToStdImage()
		{
			assertCuda(cudaGetDeviceCount(&devices_));
		}

		CUDAToStdImage::~CUDAToStdImage()
		{
		}

		void CUDAToStdImage::process(CUDAToStdImage::input_type&& img)
		{
			if(!img.valid())
			{
				finish();
				return;
			}

			for(int i = 0; i < devices_; ++i)
			{
				if(img.device() == i)
					processor_threads_.emplace_back(&CUDAToStdImage::processor, this, std::move(img), i);
			}
		}

		CUDAToStdImage::output_type CUDAToStdImage::wait()
		{
			return results_.take();
		}

		void CUDAToStdImage::processor(CUDAToStdImage::input_type&& img, int device)
		{
			assertCuda(cudaSetDevice(device));
			std::size_t hostPitch;
			std::unique_ptr<float, typename output_type::deleter_type>
				host_buffer(CUDAHostAllocator<float>::allocate(img.width(), img.height(), &hostPitch));

			assertCuda(cudaMemcpy2D(host_buffer.get(), hostPitch,
									img.data(), img.pitch(),
									img.width() * sizeof(float), img.height(),
									cudaMemcpyDeviceToHost));

			output_type result(img.width(), img.height(), std::move(host_buffer));
			result.pitch(hostPitch);
			results_.push(std::move(result));
		}

		void CUDAToStdImage::finish()
		{
#ifdef DDAFA_DEBUG
			std::cout << "CUDAToStdImage: Received poisonous pill, called finish()" << std::endl;
#endif
			for(auto&& t : processor_threads_)
				t.join();

			results_.push(output_type());
		}
	}
}
