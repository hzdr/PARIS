#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

#define BOOST_ALL_DYN_LINK
#include <boost/log/trivial.hpp>

#include <ddrf/cuda/Check.h>
#include <ddrf/cuda/Coordinates.h>
#include <ddrf/cuda/Launch.h>
#include <ddrf/cuda/Memory.h>

#include "ToHostImage.h"

namespace ddafa
{
	namespace cuda
	{
		ToHostImage::ToHostImage()
		{
			ddrf::cuda::check(cudaGetDeviceCount(&devices_));
		}

		ToHostImage::~ToHostImage()
		{
			// this is the last stage in the pipeline, time to clean up CUDA
			cudaDeviceReset();
		}

		auto ToHostImage::process(input_type&& img) -> void
		{
			if(!img.valid())
			{
				finish();
				return;
			}

			for(auto i = 0; i < devices_; ++i)
			{
				if(img.device() == i)
					processor_threads_.emplace_back(&ToHostImage::processor, this, std::move(img));
			}
		}

		auto ToHostImage::wait() -> output_type
		{
			return results_.take();
		}

		auto ToHostImage::processor(input_type&& img) -> void
		{
			BOOST_LOG_TRIVIAL(debug) << "cuda::ToHostImage: downloading from device #" << img.device();
			ddrf::cuda::check(cudaSetDevice(img.device()));

			auto host_buffer = ddrf::cuda::make_host_ptr<float>(img.width(), img.height());
			host_buffer = img.container();

			auto result = output_type(img.width(), img.height(), std::move(host_buffer));
			results_.push(std::move(result));
		}

		auto ToHostImage::finish() -> void
		{
			BOOST_LOG_TRIVIAL(debug) << "cuda::ToHostImage: Received poisonous pill, called finish()";
			for(auto&& t : processor_threads_)
				t.join();

			results_.push(output_type());
		}
	}
}
