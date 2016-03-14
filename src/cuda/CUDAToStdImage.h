/*
 * CUDAToStdImage.h
 *
 *  Created on: 01.12.2015
 *      Author: Jan Stephan
 *
 *      Converts CUDAImage to StdImage
 */

#ifndef CUDATOSTDIMAGE_H_
#define CUDATOSTDIMAGE_H_

#include <thread>
#include <vector>

#include <ddrf/Image.h>
#include <ddrf/Queue.h>
#include <ddrf/cuda/HostMemoryManager.h>
#include <ddrf/cuda/Image.h>
#include <ddrf/cuda/Memory.h>
#include <ddrf/default/Image.h>

namespace ddafa
{
	namespace impl
	{
		class CUDAToStdImage
		{
			public:
				using input_type = ddrf::Image<ddrf::cuda::Image<float>>;
				using output_type = ddrf::Image<ddrf::def::Image<float, ddrf::cuda::HostMemoryManager<float>>>;

				CUDAToStdImage();
				auto process(input_type&&) -> void;
				auto wait() -> output_type;

			protected:
				~CUDAToStdImage();

			private:
				auto processor(input_type&&, int) -> void;
				auto finish() -> void;

			private:
				ddrf::Queue<output_type> results_;
				std::vector<std::thread> processor_threads_;
				int devices_;

		};
	}
}


#endif /* CUDATOSTDIMAGE_H_ */
