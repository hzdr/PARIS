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
			std::unique_ptr<float> host_buffer(new float[img.width() * img.height()]);
			std::size_t size = img.width() * img.height() * sizeof(float);

			assertCuda(cudaMemcpy(host_buffer.get(), img.data(), size, cudaMemcpyDeviceToHost));

			results_.push(output_type(img.width(), img.height(), std::move(host_buffer)));
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
