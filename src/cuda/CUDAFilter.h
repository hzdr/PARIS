/*
 * CUDAFilter.h
 *
 *  Created on: 03.12.2015
 *      Author: Jan Stephan
 *
 *      CUDAFilter takes a weighted projection and applies a filter to it.
 */

#ifndef CUDAFILTER_H_
#define CUDAFILTER_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

#include "../common/Geometry.h"
#include "../common/Queue.h"
#include "../image/Image.h"

#include "CUDADeviceDeleter.h"
#include "CUDAImage.h"

namespace ddafa
{
	namespace impl
	{
		class CUDAFilter
		{
			public:
				using input_type = ddafa::image::Image<float, CUDAImage<float>>;
				using output_type = ddafa::image::Image<float, CUDAImage<float>>;

			public:
				CUDAFilter(ddafa::common::Geometry&& geo);
				void process(input_type&& img);
				output_type wait();

			protected:
				~CUDAFilter();

			private:
				void filterProcessor(float* buffer, std::int32_t* j_buffer, int device);
				void processor(input_type&& img, int device);
				void finish();

			private:
				ddafa::common::Queue<output_type> results_;
				std::vector<std::thread> processor_threads_;
				int devices_;
				std::size_t filter_length_;
				std::vector<std::unique_ptr<float[], CUDADeviceDeleter>> rs_;
				float tau_;
		};
	}
}


#endif /* CUDAFILTER_H_ */
