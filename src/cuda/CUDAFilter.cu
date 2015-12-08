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
			if(j >= size)
				return;

			/*
			 * r(j) with j = [ -(filter_length - 2)/2, ..., 0, ..., filter_length/2 ]
			 * tau = horizontal pixel distance
			 *
			 * 			1/8 * 1/tau^2						j = 0
			 * r(j) = {	0									j even
			 * 			-(1 / (2 * j^2 * pi^2 * tau^2))		j odd
			 */
			r[j] = (j == size / 2) ?							// is j = 0?
						(1.f / 8.f) * (1.f / powf(tau, 2.f)) 	// j = 0
						: ((j % 2 == 0) ?						// j != 0, is j even?
											0.f						// j even
											: (-1.f /				// j odd
												(2.f * powf(j, 2.f) * powf(M_PI, 2.f) * powf(tau, 2.f))));

			__syncthreads();
		}

		// info struct for the convertProjectionCallback function
		struct projectionConversionInfo
		{
			unsigned int width;
			std::size_t filter_length;
		};

		/* cuFFT loads this callback function on a per-element basis. This function is called before
		 * the actual FFT transform is executed. It fills up a projection line with zeroes for the coordinates
		 * larger than the projection width
		 */
		__device__ cufftReal convertProjectionCallback(void *dataIn, std::size_t offset, void *callerInfo,
														void *sharedPtr)
		{
			float element = ((float*) dataIn)[offset];
			auto info = (projectionConversionInfo*) callerInfo;

			std::size_t column = offset % info->filter_length;

			if(column >= info->width)
				return cufftReal(0);
			else
				return cufftReal(element);
		}
		__device__ cufftCallbackLoadR device_convert_projection_callback_ptr = convertProjectionCallback;

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
				assertCuda(cudaMalloc(&dev_buffer, filter_length_));

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
				assertCuda(cudaSetDevice(i));
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
			// allocate memory
			cufftComplex *transformed_raw;
			assertCuda(cudaMalloc(&transformed_raw, sizeof(cufftComplex) * filter_length_ * img.height()));
			std::unique_ptr<cufftComplex, CUDADeleter> transformed(transformed_raw);

			// set up cuFFT
			cufftHandle projectionPlan;
			assertCufft(cufftCreate(&projectionPlan));

			std::size_t workSizes[1];
			assertCufft(cufftMakePlan2d(projectionPlan, filter_length_, img.height(), CUFFT_R2C, workSizes));

			cufftCallbackLoadR host_convert_projection_callback_ptr;
			assertCuda(cudaMemcpyFromSymbol(&host_convert_projection_callback_ptr,
										device_convert_projection_callback_ptr,
										sizeof(host_convert_projection_callback_ptr)));

			projectionConversionInfo info = { img.width(), filter_length_ };
			assertCufft(cufftXtSetCallback(projectionPlan,
										(void **) &host_convert_projection_callback_ptr,
										CUFFT_CB_LD_REAL,
										(void **) &info));

			assertCufft(cufftExecR2C(projectionPlan, (cufftReal*) img.data(), transformed.get()));

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
