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

#include "CUDAHostAllocator.h"
#include "CUDAHostDeleter.h"
#include "CUDAImage.h"
#include "../common/Queue.h"
#include "../image/Image.h"
#include "../image/StdImage.h"

namespace ddafa
{
	namespace impl
	{
		class CUDAToStdImage : public CUDAHostAllocator<float>
		{
			public:
				using input_type = ddafa::image::Image<float, CUDAImage<float>>;
				using output_type = ddafa::image::Image<float, StdImage<float, CUDAHostAllocator<float>, CUDAHostDeleter>>;

				CUDAToStdImage();
				auto process(input_type&&) -> void;
				auto wait() -> output_type;

			protected:
				~CUDAToStdImage() = default;

			private:
				auto processor(input_type&&, int) -> void;
				auto finish() -> void;

			private:
				ddafa::common::Queue<output_type> results_;
				std::vector<std::thread> processor_threads_;
				int devices_;

		};
	}
}


#endif /* CUDATOSTDIMAGE_H_ */
