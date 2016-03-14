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

#define BOOST_ALL_DYN_LINK
#include <boost/log/trivial.hpp>

#include <ddrf/cuda/Check.h>
#include <ddrf/cuda/Memory.h>

#include "CUDAToStdImage.h"

namespace ddafa
{
	namespace impl
	{
		CUDAToStdImage::CUDAToStdImage()
		{
			ddrf::cuda::check(cudaGetDeviceCount(&devices_));
		}

		CUDAToStdImage::~CUDAToStdImage()
		{
			// this is the last stage in the pipeline, time to clean up CUDA
			cudaDeviceReset();
		}

		auto CUDAToStdImage::process(CUDAToStdImage::input_type&& img) -> void
		{
			if(!img.valid())
			{
				finish();
				return;
			}

			for(auto i = 0; i < devices_; ++i)
			{
				if(img.device() == i)
					processor_threads_.emplace_back(&CUDAToStdImage::processor, this, std::move(img), i);
			}
		}

		auto CUDAToStdImage::wait() -> CUDAToStdImage::output_type
		{
			return results_.take();
		}

		auto CUDAToStdImage::processor(CUDAToStdImage::input_type&& img, int device) -> void
		{
			ddrf::cuda::check(cudaSetDevice(device));
			auto host_buffer = ddrf::cuda::make_host_ptr<float>(img.width(), img.height());
			host_buffer = img.container();

			auto result = output_type(img.width(), img.height(), std::move(host_buffer));
			results_.push(std::move(result));
		}

		auto CUDAToStdImage::finish() -> void
		{
			BOOST_LOG_TRIVIAL(debug) << "CUDAToStdImage: Received poisonous pill, called finish()";
			for(auto&& t : processor_threads_)
				t.join();

			results_.push(output_type());
		}
	}
}
