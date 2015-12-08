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
#ifdef DDAFA_DEBUG
#include <iostream>
#endif
#include <stdexcept>
#include <string>
#include <utility>

#include "CUDACommon.h"
#include "CUDADeleter.h"
#include "CUDAWeighting.h"

#include "../common/Geometry.h"
#include "../image/Image.h"

namespace ddafa
{
	namespace impl
	{
		__global__ void weight(float* img, unsigned int width, unsigned int height,
								float h_min, float v_min, float d_dist,
								float pixel_size_horiz, float pixel_size_vert)
		{
			int j = blockIdx.x * blockDim.x + threadIdx.x; // row index
			int i = blockIdx.y * blockDim.y + threadIdx.y; // column index

			if((j < width) && (i < height))
			{
				int idx = j + i * width; // current pixel

				// detector coordinates
				float h_j = (pixel_size_horiz / 2) + j * pixel_size_horiz + h_min;
				float v_i = (pixel_size_vert / 2) + i * pixel_size_vert + v_min;

				// calculate weight
				float w_ij = d_dist * rsqrtf(powf(d_dist, 2) + powf(h_j, 2) + powf(v_i, 2));

				// apply
				img[idx] = img[idx] * w_ij;
			}
			__syncthreads();
		}

		CUDAWeighting::CUDAWeighting(ddafa::common::Geometry&& geo)
		: geo_(geo)
		, h_min_{-geo.det_offset_horiz - ((geo.det_pixels_row * geo.det_pixel_size_horiz) / 2)}
		, v_min_{-geo.det_offset_vert - ((geo.det_pixel_column * geo.det_pixel_size_vert) / 2)}
		, d_dist_{307.5} //FIXME: Remove fixed value
		{
			assertCuda(cudaGetDeviceCount(&devices_));
		}

		CUDAWeighting::~CUDAWeighting()
		{
		}

		void CUDAWeighting::process(CUDAWeighting::input_type&& img)
		{
			if(!img.valid())
			{
				// received poisonous pill, time to die
				finish();
				return;
			}

			for(int i = 0; i < devices_; ++i)
			{
#ifdef DDAFA_DEBUG
				std::cout << "CUDAWeighting: Copying to device #" << i << std::endl;
#endif
				// copy image to device
				assertCuda(cudaSetDevice(i));
				float* dev_buffer;
				std::size_t size = img.width() * img.height() * sizeof(float);
				assertCuda(cudaMalloc(&dev_buffer, size));

#ifdef DDAFA_DEBUG
				std::cout << "CUDAWeighting: Image dimensions are " << img.width()
						<< "x" << img.height() << std::endl;
				std::cout << "Size on device: " << size << " bytes" << std::endl;
#endif

				assertCuda(cudaMemcpy(dev_buffer, img.data(), size, cudaMemcpyHostToDevice));

				// execute kernel
				processor_threads_.emplace_back(&CUDAWeighting::processor, this, dev_buffer, size,
												img.width(), img.height(), i);
			}
		}

		CUDAWeighting::output_type CUDAWeighting::wait()
		{
			return results_.take();
		}

		void CUDAWeighting::processor(float* buffer, std::size_t size, std::uint32_t width, std::uint32_t height, int device)
		{
			assertCuda(cudaSetDevice(device));
#ifdef DDAFA_DEBUG
			std::cout << "CUDAWeighting: processing on device #" << device << std::endl;
#endif

			launch2D(width, height, weight, buffer, width, height, h_min_, v_min_, d_dist_,
					geo_.det_pixel_size_horiz, geo_.det_pixel_size_vert);

			output_type result(width, height, std::unique_ptr<float, CUDADeleter>(buffer));
			result.setDevice(device);

			results_.push(std::move(result));
		}

		void CUDAWeighting::finish()
		{
#ifdef DDAFA_DEBUG
			std::cout << "CUDAWeighting: Received poisonous pill, called finish()" << std::endl;
#endif
			for(auto&& t : processor_threads_)
				t.join();

#ifdef DDAFA_DEBUG
			std::cout << "CUDAWeighting: All threads joined." << std::endl;
#endif

			results_.push(output_type());
		}
	}
}
