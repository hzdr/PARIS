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
			assertCuda(cudaSetDevice(device));
			auto hostPitch = std::size_t{};

			auto host_buffer =
					std::unique_ptr<float, typename output_type::deleter_type>{
						CUDAHostAllocator<float>::allocate(img.width(), img.height(), &hostPitch)
			};

			assertCuda(cudaMemcpy2D(host_buffer.get(), hostPitch,
									img.data(), img.pitch(),
									img.width() * sizeof(float), img.height(),
									cudaMemcpyDeviceToHost));

			auto result = output_type(img.width(), img.height(), std::move(host_buffer));
			result.pitch(hostPitch);
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
