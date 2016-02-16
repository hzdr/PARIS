/*
 * CUDAWeighting.cu
 *
 *  Created on: 19.11.2015
 *      Author: Jan Stephan
 *
 *      CUDAWeighting manages the concrete implementation of weighting the projections. Implementation file.
 */

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

#define BOOST_ALL_DYN_LINK
#include <boost/log/trivial.hpp>

#include "CUDAAssert.h"
#include "CUDACommon.h"
#include "CUDADeviceDeleter.h"
#include "CUDAWeighting.h"

#include "../common/Geometry.h"
#include "../image/Image.h"

namespace ddafa
{
	namespace impl
	{
		__global__ void weight(float* img,
								std::size_t width, std::size_t height, std::size_t pitch,
								float h_min, float v_min, float d_dist,
								float pixel_size_horiz, float pixel_size_vert)
		{
			int j = getX(); // column index
			int i = getY(); // row index

			if((j < width) && (i < height))
			{
				float* row = reinterpret_cast<float*>(reinterpret_cast<char*>(img) + i * pitch);

				// detector coordinates
				float h_j = (pixel_size_horiz / 2) + j * pixel_size_horiz + h_min;
				float v_i = (pixel_size_vert / 2) + i * pixel_size_vert + v_min;

				// calculate weight
				float w_ij = d_dist * rsqrtf(powf(d_dist, 2) + powf(h_j, 2) + powf(v_i, 2));

				// apply
				row[j] = row[j] * w_ij;
			}
			__syncthreads();
		}

		CUDAWeighting::CUDAWeighting(const ddafa::common::Geometry& geo)
		: geo_(geo)
		, h_min_{-(geo.det_offset_horiz * geo.det_pixel_size_horiz) - ((static_cast<float>(geo.det_pixels_row) * geo.det_pixel_size_horiz) / 2)}
		, v_min_{-(geo.det_offset_vert * geo.det_pixel_size_vert) - ((static_cast<float>(geo.det_pixels_column) * geo.det_pixel_size_vert) / 2)}
		, d_dist_{geo.dist_det + geo.dist_src}
		{
			assertCuda(cudaGetDeviceCount(&devices_));
		}

		auto CUDAWeighting::process(CUDAWeighting::input_type&& img) -> void
		{
			if(!img.valid())
			{
				// received poisonous pill, time to die
				finish();
				return;
			}

			for(auto i = 0; i < devices_; ++i)
			{
				// execute kernel
				processor_threads_.emplace_back(&CUDAWeighting::processor, this, img, i);
			}
		}

		auto CUDAWeighting::wait() -> CUDAWeighting::output_type
		{
			return results_.take();
		}

		auto CUDAWeighting::processor(const CUDAWeighting::input_type& img, int device) -> void
		{
			assertCuda(cudaSetDevice(device));
			BOOST_LOG_TRIVIAL(debug) << "CUDAWeighting: processing on device #" << device;

			auto result = copyToDevice(img);
			result.setDevice(device);
			launch2D(result.width(), result.height(),
					weight,
					result.data(), result.width(), result.height(), result.pitch(), h_min_, v_min_, d_dist_,
					geo_.det_pixel_size_horiz, geo_.det_pixel_size_vert);
			assertCuda(cudaStreamSynchronize(0));
			results_.push(std::move(result));
		}

		auto CUDAWeighting::finish() -> void
		{
			BOOST_LOG_TRIVIAL(debug) << "CUDAWeighting: Received poisonous pill, called finish()";

			for(auto&& t : processor_threads_)
				t.join();

			results_.push(output_type());
		}

		auto CUDAWeighting::copyToDevice(const CUDAWeighting::input_type& img) -> CUDAWeighting::output_type
		{
			auto dev_buffer = static_cast<float*>(nullptr);
			auto pitch = std::size_t{};
			auto host_pitch = img.width() * sizeof(float);
			assertCuda(cudaMallocPitch(&dev_buffer, &pitch, img.width() * sizeof(float), img.height()));
			assertCuda(cudaMemcpy2D(dev_buffer, pitch,
									img.data(), host_pitch,
									img.width() * sizeof(float), img.height(),
									cudaMemcpyHostToDevice));
			auto ret = output_type(img.width(), img.height(), std::unique_ptr<float, CUDADeviceDeleter>(dev_buffer));
			ret.pitch(pitch);

			return ret;
		}
	}
}
