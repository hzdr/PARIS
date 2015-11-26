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
#include <thread>
#include <utility>
#include <vector>

#include "CUDACommon.h"
#include "CUDAWeighting.h"

#include "../image/Image.h"

namespace ddafa
{
	namespace impl
	{
		__global__ void weight(float* img, unsigned width, unsigned height,
								float h_min, float v_min, float d_dist,
								float pixel_size_horiz, float pixel_size_vert)
		{
			int j = blockIdx.x * blockDim.x + threadIdx.x; // row index
			int i = blockIdx.y * blockDim.y + threadIdx.y; // column index

			if((j >= width) || (i >= height))
				return;

			int idx = i + j * width; // current pixel

			// detector coordinates
			float h_j = (pixel_size_horiz / 2) + j * pixel_size_horiz + h_min;
			float v_i = (pixel_size_vert / 2) + i * pixel_size_vert + v_min;

			// calculate weight
			float w_ij = d_dist * rsqrtf(powf(d_dist, 2) + powf(h_j, 2) + powf(v_i, 2));

			// apply
			img[idx] = img[idx] * w_ij;
		}

		CUDAWeighting::CUDAWeighting(ddafa::common::Geometry geo)
		: geo_(geo)
		, h_min_{-geo.det_offset_horiz - ((geo.det_pixels_row * geo.det_pixel_size_horiz) / 2)}
		, v_min_{-geo.det_offset_vert - ((geo.det_pixel_column * geo.det_pixel_size_vert) / 2)}
		, d_dist_{geo.dist_det - geo.dist_src}
		{
			cudaError_t err = cudaGetDeviceCount(&devices_);

			switch(err)
			{
				case cudaSuccess:
					break;

				case cudaErrorNoDevice:
					throw std::runtime_error("CUDAWeighting: No CUDA devices found.");

				case cudaErrorInsufficientDriver:
					throw std::runtime_error("CUDAWeighting: Insufficient driver.");
			}
		}

		CUDAWeighting::~CUDAWeighting()
		{
		}

		void CUDAWeighting::process(CUDAWeighting::input_type&& img)
		{
			if(!img.valid())
			{
				// received poisonous pill, time to die
				results_.push(output_type());
				return;
			}

			std::vector<std::thread> processor_threads;
			for(int i = 0; i < devices_; ++i)
			{
				// copy image to device
				cudaSetDevice(i);
				float* dev_buffer;
				std::size_t size = img.width() * img.height() * sizeof(float);
				cudaError_t err = cudaMalloc(&dev_buffer, size);

				switch(err)
				{
					case cudaErrorMemoryAllocation:
						throw std::runtime_error("CUDAWeighting: Error while allocating memory");

					case cudaSuccess:
						default:
						break;
				}

				err = cudaMemcpy(dev_buffer, img.data(), size, cudaMemcpyHostToDevice);
				switch(err)
				{
					case cudaErrorInvalidValue:
						throw std::runtime_error("CUDAWeighting: Invalid value");

					case cudaErrorInvalidDevicePointer:
						throw std::runtime_error("CUDAWeighting: Invalid device pointer");

					case cudaErrorInvalidMemcpyDirection:
						throw std::runtime_error("CUDAWeighting: Invalid memcpy direction");

					case cudaSuccess:
						default:
						break;
				}
				// execute kernel
				processor_threads.emplace_back(&CUDAWeighting::processor, this, dev_buffer, size,
												img.width(), img.height());
			}

			for(auto&& t : processor_threads)
				t.join();
		}

		CUDAWeighting::output_type CUDAWeighting::wait()
		{
			return results_.take();
		}

		void CUDAWeighting::processor(float* buffer, std::size_t size, std::uint32_t width, std::uint32_t height)
		{
			launch(size, weight, buffer, width, height, h_min_, v_min_, d_dist_,
					geo_.det_pixel_size_horiz, geo_.det_pixel_size_vert);
			output_type result(width, height, buffer);

			int device;
			cudaGetDevice(&device);

			result.setDevice(device);

			results_.push(std::move(result));
		}
	}
}
