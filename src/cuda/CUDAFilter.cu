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
#include "CUDADeviceDeleter.h"
#include "CUDAFilter.h"

namespace ddafa
{
	namespace impl
	{
		__global__ void createFilter(float* __restrict__ r, const int* __restrict__ j,
				std::size_t size, float tau)
		{
			int x = getX();

			/*
			 * r(j) with j = [ -(filter_length - 2)/2, ..., 0, ..., filter_length/2 ]
			 * tau = horizontal pixel distance
			 *
			 * 			1/8 * 1/tau^2						j = 0
			 * r(j) = {	0									j even
			 * 			-(1 / (2 * j^2 * pi^2 * tau^2))		j odd
			 *
			 */
			if(x < size)
			{
				if(j[x] == 0) // is j = 0?
					r[x] = (1.f / 8.f) * (1.f / powf(tau, 2)); // j = 0
				else // j != 0
				{
					if(j[x] % 2 == 0) // is j even?
						r[x] = 0.f; // j is even
					else // j is odd
						r[x] = (-1.f / (2.f * powf(j[x], 2) * powf(M_PI, 2) * powf(tau, 2)));

				}
			}
			__syncthreads();
		}

		__global__ void convertProjection(float* __restrict__ output, const float* __restrict__ input,
				unsigned int width, unsigned int height, std::size_t filter_length)
		{
			int x = getX();
			int y = getY();

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

		__global__ void convertFiltered(float* __restrict__ output, const float* __restrict__ input,
				unsigned int width, unsigned int height, std::size_t filter_length)
		{
			int x = getX();
			int y = getY();

			if((x < width) && (y < height)) {
				int idx = x + y * width;
				output[idx] = input[idx] / filter_length;
			}
			__syncthreads();
		}

		__global__ void createK(cufftComplex* __restrict__ data, std::size_t filter_length, float tau)
		{
			int x = getX();
			if(x < filter_length)
			{
				float result = tau * fabsf(sqrtf(powf(data[x].x, 2.f) + powf(data[x].y, 2.f)));
				data[x].x = result;
				data[x].y = result;
			}
			__syncthreads();
		}

		__global__ void applyFilter(cufftComplex* __restrict__ data, const cufftComplex* __restrict__ filter,
				std::size_t filter_length, std::uint32_t data_height)
		{
			int x = getX();
			int y = getY();

			if((x < filter_length) && (y < data_height))
			{
				int idx = x + y * filter_length;

				float a1, b1, k1, k2;
				a1 = data[idx].x;
				b1 = data[idx].y;
				k1 = filter[x].x;
				k2 = filter[x].y;

				data[idx].x = a1 * k1;
				data[idx].y = b1 * k2;
			}
			__syncthreads();
		}

		CUDAFilter::CUDAFilter(const ddafa::common::Geometry& geo)
		: filter_length_{static_cast<decltype(filter_length_)>(
				2 * std::pow(2, std::ceil(std::log2(float(geo.det_pixels_column))))
				)}
		, tau_{geo.det_pixel_size_horiz}
		{
			assertCuda(cudaGetDeviceCount(&devices_));

			rs_.resize(devices_);

			std::vector<std::thread> filter_creation_threads;
			for(int i = 0; i < devices_; ++i)
			{
				filter_creation_threads.emplace_back(&CUDAFilter::filterProcessor, this, i);
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

		void CUDAFilter::filterProcessor(int device)
		{
			assertCuda(cudaSetDevice(device));
#ifdef DDAFA_DEBUG
			std::cout << "CUDAFilter: Creating filter on device #" << device << std::endl;
#endif
			float* buffer_raw;
			assertCuda(cudaMalloc(&buffer_raw, filter_length_ * sizeof(float)));
			std::unique_ptr<float[], CUDADeviceDeleter> buffer(buffer_raw);

			// see documentation in kernel createFilter for explanation
			std::int32_t j_host_buffer[filter_length_];
			auto filter_length_signed = static_cast<std::int32_t>(filter_length_);
			std::int32_t j = -((filter_length_signed - 2) / 2);
			for(std::size_t k = 0; k <= (filter_length_); ++k, ++j)
				j_host_buffer[k] = j;

			std::int32_t* j_dev_buffer_raw;
			assertCuda(cudaMalloc(&j_dev_buffer_raw, filter_length_ * sizeof(std::int32_t)));
			std::unique_ptr<std::int32_t[], CUDADeviceDeleter> j_dev_buffer(j_dev_buffer_raw);
			assertCuda(cudaMemcpy(j_dev_buffer.get(), j_host_buffer, filter_length_ * sizeof(std::int32_t),
									cudaMemcpyHostToDevice));

			launch1D(filter_length_,
					createFilter,
					buffer.get(), static_cast<const std::int32_t*>(j_dev_buffer.get()),
					filter_length_, tau_);
			rs_[device] = std::move(buffer);
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
			std::unique_ptr<float, CUDADeviceDeleter> converted(converted_raw);
			launch2D(filter_length_, img.height(), convertProjection, converted.get(),
					static_cast<const float*>(img.data()), img.width(), img.height(), filter_length_);

			// allocate memory
			std::size_t transformed_filter_length = filter_length_ / 2 + 1; // filter_length_ is always a power of 2
			cufftComplex* transformed_raw;
			assertCuda(cudaMalloc(&transformed_raw,
					sizeof(cufftComplex) * transformed_filter_length * img.height()));
			std::unique_ptr<cufftComplex, CUDADeviceDeleter> transformed(transformed_raw);

			cufftComplex* filter_raw;
			assertCuda(cudaMalloc(&filter_raw,
					sizeof(cufftComplex) * transformed_filter_length));
			std::unique_ptr<cufftComplex, CUDADeviceDeleter> filter(filter_raw);

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
			assertCufft(cufftExecR2C(filterPlan, static_cast<cufftReal*>(rs_[device].get()), filter.get()));

			// create K
			launch1D(transformed_filter_length, createK, filter.get(), transformed_filter_length, tau_);

			// multiply the results
			launch2D(transformed_filter_length, img.height(), applyFilter, transformed.get(),
				static_cast<const cufftComplex*>(filter.get()), transformed_filter_length, img.height());

			// run inverse FFT -- note that C2R transformations are implicitly inverse
			assertCufft(cufftExecC2R(inversePlan, transformed.get(), static_cast<cufftReal*>(converted.get())));

			// convert back to image dimensions and normalize
			launch2D(filter_length_, img.height(), convertFiltered, img.data(),
					static_cast<const float*>(converted.get()),	img.width(), img.height(), filter_length_);

			// clean up
			assertCufft(cufftDestroy(inversePlan));
			assertCufft(cufftDestroy(filterPlan));
			assertCufft(cufftDestroy(projectionPlan));

			assertCuda(cudaStreamSynchronize(0));

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
