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

#include "CUDAImage.h"
#include "../common/Queue.h"
#include "../image/Image.h"
#include "../image/StdImage.h"

namespace ddafa
{
	namespace impl
	{
		class CUDAToStdImage
		{
			public:
				using input_type = ddafa::image::Image<float, CUDAImage<float>>;
				using output_type = ddafa::image::Image<float, StdImage<float>>;

				CUDAToStdImage();
				void process(input_type&& img);
				output_type wait();

			protected:
				~CUDAToStdImage();

			private:
				void processor(input_type&& img, int device);
				void finish();

			private:
				ddafa::common::Queue<output_type> results_;
				std::vector<std::thread> processor_threads_;
				int devices_;

		};
	}
}


#endif /* CUDATOSTDIMAGE_H_ */
