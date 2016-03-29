#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

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
			CHECK(cudaGetDeviceCount(&devices_));
			for(auto i = 0; i < devices_; ++i)
				processor_threads_[i] = std::thread{&Weighting::processor, this, i};
		}

		auto Weighting::process(input_type&& img) -> void
		{
			if(img.valid())
				map_imgs_[img.device()].push(std::move(img));
			else
			{
				BOOST_LOG_TRIVIAL(debug) << "cuda::Weighting: Received poisonous pill, finishing...";
				for(auto i = 0; i < devices_; ++i)
					map_imgs_[i].push(input_type());

				for(auto i = 0; i < devices_; ++i)
					processor_threads_[i].join();

				results_.push(output_type());
				BOOST_LOG_TRIVIAL(info) << "cuda::Weighting: Done.";
			}
		}

		auto Weighting::wait() -> output_type
		{
			return results_.take();
		}

		auto Weighting::processor(int device) -> void
		{
			CHECK(cudaSetDevice(device));
			while(true)
			{
				auto img = map_imgs_[device].take();
				if(!img.valid())
					break;

				BOOST_LOG_TRIVIAL(debug) << "cuda::Weighting: processing image #" << img.index() << " on device #" << device;

				ddrf::cuda::launch(img.width(), img.height(),
						weight,
						img.data(), img.width(), img.height(), img.pitch(), h_min_, v_min_, d_dist_,
						geo_.det_pixel_size_horiz, geo_.det_pixel_size_vert);

				CHECK(cudaStreamSynchronize(0));
				results_.push(std::move(img));
			}
		}
	}
}
