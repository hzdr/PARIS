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

#include <thread>
#include <vector>

#include "../common/Queue.h"
#include "../image/Image.h"

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
				CUDAFilter();
				void process(input_type&& img);
				output_type wait();

			protected:
				~CUDAFilter();

			private:
				void processor();
				void finish();

			private:
				ddafa::common::Queue<output_type> results_;
				std::vector<std::thread> processor_threads_;
				int devices_;
		};
	}
}


#endif /* CUDAFILTER_H_ */
