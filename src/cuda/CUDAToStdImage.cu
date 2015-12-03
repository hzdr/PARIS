/*
 * CUDAToStdImage.cu
 *
 *  Created on: 01.12.2015
 *      Author: Jan Stephan
 *
 *      Converts CUDAImage to StdImage. Implementation file.
 */

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "CUDAToStdImage.h"

namespace ddafa
{
	namespace impl
	{
		CUDAToStdImage::CUDAToStdImage()
		{
			cudaError_t err = cudaGetDeviceCount(&devices_);
			if(err != cudaSuccess)
				throw std::runtime_error("CUDAToStdImage::CUDAToStdImage: " + std::string(cudaGetErrorString(err)));
		}

		CUDAToStdImage::~CUDAToStdImage()
		{
		}

		void CUDAToStdImage::process(CUDAToStdImage::input_type&& img)
		{
			if(!img.valid())
			{
				results_.push(output_type());
				return;
			}

			std::vector<std::thread> processor_threads;
			for(int i = 0; i < devices_; ++i)
			{
				cudaSetDevice(i);
				if(img.device() == i)
					processor_threads.emplace_back(&CUDAToStdImage::processor, this, std::move(img));
			}

			for(auto&& t : processor_threads)
				t.join();
		}

		CUDAToStdImage::output_type CUDAToStdImage::wait()
		{
			return results_.take();
		}

		void CUDAToStdImage::processor(CUDAToStdImage::input_type&& img)
		{
			std::unique_ptr<float> host_buffer(new float[img.width() * img.height()]);
			std::size_t size = img.width() * img.height() * sizeof(float);

			cudaError_t err = cudaMemcpy(host_buffer.get(), img.data(), size, cudaMemcpyDeviceToHost);
			if(err != cudaSuccess)
				throw std::runtime_error("CUDAToStdImage::processor: " + std::string(cudaGetErrorString(err)));

			results_.push(output_type(img.width(), img.height(), std::move(host_buffer)));
		}
	}
}
