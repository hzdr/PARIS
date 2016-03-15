#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

#define BOOST_ALL_DYN_LINK
#include <boost/log/trivial.hpp>

#include <ddrf/Image.h>
#include <ddrf/cuda/Check.h>
#include <ddrf/cuda/Coordinates.h>
#include <ddrf/cuda/Launch.h>

#include "Weighting.h"

#include "../common/Geometry.h"

namespace ddafa
{
	namespace cuda
	{
		__global__ void weight(float* img,
								std::size_t width, std::size_t height, std::size_t pitch,
								float h_min, float v_min, float d_dist,
								float pixel_size_horiz, float pixel_size_vert)
		{
			auto j = ddrf::cuda::getX(); // column index
			auto i = ddrf::cuda::getY(); // row index

			if((j < width) && (i < height))
			{
				auto* row = reinterpret_cast<float*>(reinterpret_cast<char*>(img) + i * pitch);

				// detector coordinates
				auto h_j = (pixel_size_horiz / 2) + j * pixel_size_horiz + h_min;
				auto v_i = (pixel_size_vert / 2) + i * pixel_size_vert + v_min;

				// calculate weight
				auto w_ij = d_dist * rsqrtf(powf(d_dist, 2) + powf(h_j, 2) + powf(v_i, 2));

				// apply
				row[j] = row[j] * w_ij;
			}
			__syncthreads();
		}

		Weighting::Weighting(const common::Geometry& geo)
		: geo_(geo)
		, h_min_{-(geo.det_offset_horiz * geo.det_pixel_size_horiz) - ((static_cast<float>(geo.det_pixels_row) * geo.det_pixel_size_horiz) / 2)}
		, v_min_{-(geo.det_offset_vert * geo.det_pixel_size_vert) - ((static_cast<float>(geo.det_pixels_column) * geo.det_pixel_size_vert) / 2)}
		, d_dist_{geo.dist_det + geo.dist_src}
		{
			ddrf::cuda::check(cudaGetDeviceCount(&devices_));
		}

		auto Weighting::process(input_type&& img) -> void
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
				processor_threads_.emplace_back(&Weighting::processor, this, img, i);
			}
		}

		auto Weighting::wait() -> output_type
		{
			return results_.take();
		}

		auto Weighting::processor(const input_type& img, int device) -> void
		{
			ddrf::cuda::check(cudaSetDevice(device));
			// BOOST_LOG_TRIVIAL(debug) << "CUDAWeighting: processing on device #" << device;

			auto result = output_type{};
			result.setDevice(device);
			result = img;

			ddrf::cuda::launch(result.width(), result.height(),
					weight,
					result.data(), result.width(), result.height(), result.pitch(), h_min_, v_min_, d_dist_,
					geo_.det_pixel_size_horiz, geo_.det_pixel_size_vert);

			ddrf::cuda::check(cudaStreamSynchronize(0));
			results_.push(std::move(result));
		}

		auto Weighting::finish() -> void
		{
			BOOST_LOG_TRIVIAL(debug) << "CUDAWeighting: Received poisonous pill, called finish()";

			for(auto&& t : processor_threads_)
				t.join();

			results_.push(output_type());
		}
	}
}
