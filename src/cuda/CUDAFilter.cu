/*
 * CUDAFilter.cu
 *
 *  Created on: 03.12.2015
 *      Author: Jan Stephan
 *
 *      CUDAFilter takes a weighted projection and applies a filter to it. Implementation file.
 */

#include <cmath>
#include <cstddef>
#include <ctgmath>
#ifdef DDAFA_DEBUG
#include <iostream>
#endif
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <cufft.h>
#include <cufftXt.h>

#include "../common/Geometry.h"

#include "CUDAAssert.h"
#include "CUDACommon.h"
#include "CUDADeleter.h"
#include "CUDAFilter.h"

namespace ddafa
{
	namespace impl
	{
		__global__ void createFilter(float* r, std::size_t size, float tau)
		{
			int j = blockIdx.x * blockDim.x + threadIdx.x;

			/*
			 * r(j) with j = [ -(filter_length - 2)/2, ..., 0, ..., filter_length/2 ]
			 * tau = horizontal pixel distance
			 *
			 * 			1/8 * 1/tau^2						j = 0
			 * r(j) = {	0									j even
			 * 			-(1 / (2 * j^2 * pi^2 * tau^2))		j odd
			 */
			if(j < size)
			{
				if(j == (size / 2) - 1) // is j = 0?
					r[j] = (1.f / 8.f) * (1.f / powf(tau, 2)); // j = 0
				else // j != 0
				{
					if(j % 2 == 0) // is j even?
						r[j] = 0.f; // j is even
					else // j is odd
						r[j] = (-1.f / (2.f * powf(j, 2) * powf(M_PI, 2) * powf(tau, 2)));
				}
			}
			__syncthreads();
		}

		__global__ void convertProjection(float* output, const float* input,
				unsigned int width, unsigned int height, std::size_t filter_length)
		{
			int x = blockIdx.x * blockDim.x + threadIdx.x;
			int y = blockIdx.y * blockDim.y + threadIdx.y;

			if((x < filter_length) && (y < height))
			{
				int idx = x + y * width;
				if(x < width)
					output[idx] = input[idx];
				else
					output[idx] = 0.0f;
			}
			__syncthreads();
		}

		__global__ void convertFiltered(float* output, const float* input,
				unsigned int width, unsigned int height, std::size_t filter_length)
		{
			int x = blockIdx.x * blockDim.x + threadIdx.x;
			int y = blockIdx.y * blockDim.y + threadIdx.y;

			if((x < width) && (y < height)) {
				int idx = x + y * width;
				output[idx] = input[idx] / (filter_length * height);
			}
			__syncthreads();
		}

		__global__ void applyFilter(cufftComplex* data, const cufftComplex* filter,
				std::size_t filter_length, std::uint32_t data_height)
		{
			int x = blockIdx.x * blockDim.x + threadIdx.x;
			int y = blockIdx.y * blockDim.y + threadIdx.y;

			if((x < filter_length) && (y < data_height))
			{
				int idx = x + y * filter_length;

				float a1, b1, a2, b2;
				a1 = data[idx].x;
				b1 = data[idx].y;
				a2 = filter[x].x;
				b2 = filter[x].y;

				data[idx].x = a1 * a2 - b1 * b2;
				data[idx].y = a1 * b2 + a2 * b1;
			}

			__syncthreads();
		}

		CUDAFilter::CUDAFilter(ddafa::common::Geometry&& geo)
		: filter_length_{static_cast<decltype(filter_length_)>(
				2 * std::pow(2, std::ceil(std::log2(float(geo.det_pixel_column))))
				)}

		{
			assertCuda(cudaGetDeviceCount(&devices_));

			auto tau = geo.det_pixel_size_horiz;
			rs_.resize(devices_);

			std::vector<std::thread> filter_creation_threads;
			for(int i = 0; i < devices_; ++i)
			{
				assertCuda(cudaSetDevice(i));
				float *dev_buffer;
				assertCuda(cudaMalloc(&dev_buffer, filter_length_ * sizeof(float)));

				filter_creation_threads.emplace_back(&CUDAFilter::filterProcessor, this, dev_buffer, tau, i);
			}

			for(auto&& t : filter_creation_threads)
				t.join();
		}

		CUDAFilter::~CUDAFilter()
		{
		}

		void CUDAFilter::process(CUDAFilter::input_type&& img)
		{
			if(!img.valid())
			{
				// received poisonous pill, time to die
				finish();
				return;
			}

			for(int i = 0; i < devices_; ++i)
			{
				if(img.device() == i)
					processor_threads_.emplace_back(&CUDAFilter::processor, this, std::move(img), i);
			}
		}

		CUDAFilter::output_type CUDAFilter::wait()
		{
			return results_.take();
		}

		void CUDAFilter::filterProcessor(float *buffer, float tau, int device)
		{
			assertCuda(cudaSetDevice(device));
#ifdef DDAFA_DEBUG
			std::cout << "CUDAFilter: Creating filter on device #" << device << std::endl;
#endif
			launch1D(filter_length_, createFilter, buffer, filter_length_, tau);
			rs_[device].reset(buffer);
		}

		void CUDAFilter::processor(CUDAFilter::input_type&& img, int device)
		{
			assertCuda(cudaSetDevice(device));

#ifdef DDAFA_DEBUG
			std::cout << "CUDAFilter: processing on device #" << device << std::endl;
#endif
			// convert projection to new dimensions
			float* converted_raw;
			assertCuda(cudaMalloc(&converted_raw, sizeof(float) * filter_length_ * img.height()));
			std::unique_ptr<float, CUDADeleter> converted(converted_raw);
			launch2D(filter_length_, img.height(), convertProjection, converted.get(),
					static_cast<const float*>(img.data()), img.width(), img.height(), filter_length_);

			// allocate memory
			std::size_t transformed_filter_length = filter_length_ / 2 + 1; // filter_length_ is always a power of 2
			cufftComplex* transformed_raw;
			assertCuda(cudaMalloc(&transformed_raw,
					sizeof(cufftComplex) * transformed_filter_length * img.height()));
			std::unique_ptr<cufftComplex, CUDADeleter> transformed(transformed_raw);

			cufftComplex* filter_raw;
			assertCuda(cudaMalloc(&filter_raw,
					sizeof(cufftComplex) * transformed_filter_length));
			std::unique_ptr<cufftComplex, CUDADeleter> filter(filter_raw);

			// set up cuFFT
			int n_proj[] = { static_cast<int>(filter_length_) };
			int n_inverse[] = { static_cast<int>(transformed_filter_length) };

			cufftHandle projectionPlan;
			assertCufft(cufftCreate(&projectionPlan));
			assertCufft(cufftSetStream(projectionPlan, 0));
			std::size_t projWorkSize;
			assertCufft(cufftMakePlanMany(projectionPlan, 1, n_proj, n_proj, 1, filter_length_,
										n_inverse, 1, transformed_filter_length, CUFFT_R2C,
										img.height(), &projWorkSize));

			cufftHandle filterPlan;
			assertCufft(cufftCreate(&filterPlan));
			assertCufft(cufftSetStream(filterPlan, 0));
			std::size_t filterWorkSize;
			assertCufft(cufftMakePlan1d(filterPlan, filter_length_, CUFFT_R2C, 1, &filterWorkSize));

			cufftHandle inversePlan;
			assertCufft(cufftCreate(&inversePlan));
			assertCufft(cufftSetStream(inversePlan, 0));
			std::size_t inverseWorkSize;
			assertCufft(cufftMakePlanMany(inversePlan, 1, n_proj, n_inverse, 1, transformed_filter_length,
										n_proj, 1, filter_length_, CUFFT_C2R, img.height(), &inverseWorkSize));

			// run the FFT for projection and filter -- note that R2C transformations are implicitly forward
			assertCufft(cufftExecR2C(projectionPlan, static_cast<cufftReal*>(converted.get()), transformed.get()));
			assertCuda(cudaStreamSynchronize(0));

			assertCufft(cufftExecR2C(filterPlan, static_cast<cufftReal*>(rs_[device].get()), filter.get()));
			assertCuda(cudaStreamSynchronize(0));

			// multiply the results
			launch2D(transformed_filter_length, img.height(), applyFilter, transformed.get(),
					(const cufftComplex*) filter.get(), transformed_filter_length, img.height());

			// run inverse FFT -- note that C2R transformations are implicitly inverse
			assertCufft(cufftExecC2R(inversePlan, transformed.get(), static_cast<cufftReal*>(converted.get())));
			assertCuda(cudaStreamSynchronize(0));

			// convert back to image dimensions and normalize
			launch2D(filter_length_, img.height(), convertFiltered, img.data(), (const float*) converted.get(), img.width(), img.height(), filter_length_);

			// clean up
			assertCufft(cufftDestroy(inversePlan));
			assertCufft(cufftDestroy(filterPlan));
			assertCufft(cufftDestroy(projectionPlan));

			results_.push(std::move(img));
		}

		void CUDAFilter::finish()
		{
#ifdef DDAFA_DEBUG
				std::cout << "CUDAFilter: Received poisonous pill, called finish()" << std::endl;
#endif

				for(auto&& t : processor_threads_)
					t.join();

				results_.push(output_type());
		}
	}
}
