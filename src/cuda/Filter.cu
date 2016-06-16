/*
 * CUDAFilter.cu
 *
 *  Created on: 03.12.2015
 *      Author: Jan Stephan
 *
 *      CUDAFilter takes a weighted projection and applies a filter to it. Implementation file.
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <ctgmath>
#include <thread>
#include <utility>
#include <vector>

#include <boost/log/trivial.hpp>

#include <cufft.h>

#include <ddrf/cuda/Check.h>
#include <ddrf/cuda/Coordinates.h>
#include <ddrf/cuda/Launch.h>
#include <ddrf/cuda/Memory.h>

#include "../common/Geometry.h"

#include "Filter.h"

namespace ddafa
{
	namespace cuda
	{
		__global__ void createFilter(float* __restrict__ r, const std::int32_t* __restrict__ j,
				std::size_t size, float tau)
		{
			auto x = ddrf::cuda::getX();

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
		}

		__global__ void convertProjection(float* __restrict__ output, const float* __restrict__ input,
				std::size_t width, std::size_t height, std::size_t input_pitch, std::size_t output_pitch,
				std::size_t filter_length)
		{
			auto x = ddrf::cuda::getX();
			auto y = ddrf::cuda::getY();

			if((x < filter_length) && (y < height))
			{
				auto output_row = reinterpret_cast<float*>(reinterpret_cast<char*>(output) + y * output_pitch);
				auto input_row = reinterpret_cast<const float*>(reinterpret_cast<const char*>(input) + y * input_pitch);

				if(x < width)
					output_row[x] = input_row[x];
				else
					output_row[x] = 0.0f;
			}
		}

		__global__ void convertFiltered(float* __restrict__ output, const float* __restrict__ input,
				std::size_t width, std::size_t height, std::size_t output_pitch,
				std::size_t input_pitch, std::size_t filter_length)
		{
			auto x = ddrf::cuda::getX();
			auto y = ddrf::cuda::getY();

			if((x < width) && (y < height)) {
				auto output_row = reinterpret_cast<float*>(reinterpret_cast<char*>(output) + y * output_pitch);
				auto input_row = reinterpret_cast<const float*>(reinterpret_cast<const char*>(input) + y * input_pitch);
				output_row[x] = input_row[x] / filter_length;
			}
		}

		__global__ void createK(cufftComplex* __restrict__ data, std::size_t filter_length, float tau)
		{
			auto x = ddrf::cuda::getX();
			if(x < filter_length)
			{
				auto result = tau * fabsf(sqrtf(powf(data[x].x, 2.f) + powf(data[x].y, 2.f)));
				data[x].x = result;
				data[x].y = result;
			}
		}

		__global__ void applyFilter(cufftComplex* __restrict__ data, const cufftComplex* __restrict__ filter,
				std::size_t filter_length, std::size_t data_height, std::size_t pitch)
		{
			auto x = ddrf::cuda::getX();
			auto y = ddrf::cuda::getY();

			if((x < filter_length) && (y < data_height))
			{
				auto row = reinterpret_cast<cufftComplex*>(reinterpret_cast<char*>(data) + y * pitch);

				auto a1 = 0.f, b1 = 0.f, k1 = 0.f, k2 = 0.f;
				a1 = row[x].x;
				b1 = row[x].y;
				k1 = filter[x].x;
				k2 = filter[x].y;

				row[x].x = a1 * k1;
				row[x].y = b1 * k2;
			}
		}

		Filter::Filter(const common::Geometry& geo)
		: filter_length_{static_cast<decltype(filter_length_)>(
				2 * std::pow(2, std::ceil(std::log2(float(geo.det_pixels_column))))
				)}
		, tau_{geo.det_pixel_size_horiz}
		{
			CHECK(cudaGetDeviceCount(&devices_));

			rs_.resize(static_cast<unsigned int>(devices_));

			auto filter_creation_threads = std::vector<std::thread>{};
			for(auto i = 0; i < devices_; ++i)
			{
				filter_creation_threads.emplace_back(&Filter::filterProcessor, this, i);
				processor_threads_[i] = std::thread{&Filter::processor, this, i};
			}

			for(auto&& t : filter_creation_threads)
				t.join();
		}

		auto Filter::process(input_type&& img) -> void
		{
			if(img.valid())
				map_imgs_[img.device()].push(std::move(img));
			else
			{
				BOOST_LOG_TRIVIAL(debug) << "cuda::Filter: Received poisonous pill, finishing...";
				for(auto i = 0; i < devices_; ++i)
					map_imgs_[i].push(input_type());

				for(auto i = 0; i < devices_; ++i)
					processor_threads_[i].join();

				results_.push(output_type());
				BOOST_LOG_TRIVIAL(info) << "cuda::Filter: Done.";
			}
		}

		auto Filter::wait() -> output_type
		{
			return results_.take();
		}

		auto Filter::filterProcessor(int device) -> void
		{
			CHECK(cudaSetDevice(device));
			BOOST_LOG_TRIVIAL(debug) << "cuda::Filter: Creating filter on device #" << device;

			auto buffer = ddrf::cuda::make_device_ptr<float>(filter_length_);

			// see documentation in kernel "createFilter" for explanation
			auto j_host_buffer = ddrf::cuda::make_host_ptr<std::int32_t>(filter_length_);
			auto filter_length_signed = static_cast<std::int32_t>(filter_length_);
			auto j = -((filter_length_signed - 2) / 2);
			std::iota(j_host_buffer.get(), j_host_buffer.get() + filter_length_, j);

			auto j_dev_buffer = ddrf::cuda::make_device_ptr<std::int32_t>(filter_length_);
			ddrf::cuda::copy_sync(j_dev_buffer, j_host_buffer);

			ddrf::cuda::launch(filter_length_,
					createFilter,
					buffer.get(), static_cast<const std::int32_t*>(j_dev_buffer.get()),
					filter_length_, tau_);
			rs_[static_cast<std::size_t>(device)] = std::move(buffer);
		}

		auto Filter::processor(int device) -> void
		{
			CHECK(cudaSetDevice(device));
			while(true)
			{
				auto img = map_imgs_[device].take();
				if(!img.valid())
					break;

				BOOST_LOG_TRIVIAL(debug) << "cuda::Filter: processing image #" << img.index() << " on device #" << device;

				// convert projection to new dimensions
				auto converted = ddrf::cuda::make_device_ptr<float>(filter_length_, img.height());
				ddrf::cuda::launch(filter_length_, img.height(), convertProjection, converted.get(),
						static_cast<const float*>(img.data()), img.width(), img.height(), img.pitch(), converted.pitch(), filter_length_);

				// allocate memory
				auto transformed_filter_length = filter_length_ / 2 + 1; // filter_length_ is always a power of 2
				auto transformed = ddrf::cuda::make_device_ptr<cufftComplex>(transformed_filter_length, img.height());

				auto filter = ddrf::cuda::make_device_ptr<cufftComplex>(transformed_filter_length);

				// set up cuFFT
				auto n_proj = std::vector<int>{ static_cast<int>(filter_length_) };
				auto proj_dist = static_cast<int>(converted.pitch() / sizeof(float));
				auto proj_nembed = std::vector<int>{ proj_dist };

				auto trans_dist = static_cast<int>(transformed.pitch() / sizeof(cufftComplex));
				auto trans_nembed = std::vector<int>{ trans_dist };

				auto projectionPlan = cufftHandle{};
				CHECK_CUFFT(cufftPlanMany(&projectionPlan, 					// plan
											1, 								// rank (dimension)
											n_proj.data(),					// input dimension size
											proj_nembed.data(),				// input storage dimensions
											1,								// distance between two successive input elements
											proj_dist,						// distance between two input signals
											trans_nembed.data(),			// output storage dimensions
											1,								// distance between two successive output elements
											trans_dist, 					// distance between two output signals
											CUFFT_R2C,						// transform data type
											static_cast<int>(img.height()) 	// batch size
											));

				auto filterPlan = cufftHandle{};
				CHECK_CUFFT(cufftPlan1d(&filterPlan, static_cast<int>(filter_length_), CUFFT_R2C, 1));

				auto inversePlan = cufftHandle{};
				CHECK_CUFFT(cufftPlanMany(&inversePlan,
											1,
											n_proj.data(),
											trans_nembed.data(),
											1,
											trans_dist,
											proj_nembed.data(),
											1,
											proj_dist,
											CUFFT_C2R,
											static_cast<int>(img.height())
											));

				// run the FFT for projection and filter -- note that R2C transformations are implicitly forward
				CHECK_CUFFT(cufftExecR2C(projectionPlan, converted.get(), transformed.get()));
				CHECK_CUFFT(cufftExecR2C(filterPlan, rs_[static_cast<std::size_t>(img.device())].get(), filter.get()));

				// create K
				ddrf::cuda::launch(transformed_filter_length, createK, filter.get(), transformed_filter_length, tau_);

				// multiply the results
				ddrf::cuda::launch(transformed_filter_length, img.height(), applyFilter, transformed.get(),
					static_cast<const cufftComplex*>(filter.get()), transformed_filter_length, img.height(), transformed.pitch());

				// run inverse FFT -- note that C2R transformations are implicitly inverse
				CHECK_CUFFT(cufftExecC2R(inversePlan, transformed.get(), converted.get()));

				// convert back to image dimensions and normalize
				ddrf::cuda::launch(filter_length_, img.height(), convertFiltered, img.data(),
						static_cast<const float*>(converted.get()),	img.width(), img.height(), img.pitch(), converted.pitch(), filter_length_);

				// clean up
				CHECK_CUFFT(cufftDestroy(inversePlan));
				CHECK_CUFFT(cufftDestroy(filterPlan));
				CHECK_CUFFT(cufftDestroy(projectionPlan));

				results_.push(std::move(img));
			}
		}
	}
}
