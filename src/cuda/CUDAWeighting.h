/*
 * CUDAWeighting.h
 *
 *  Created on: 19.11.2015
 *      Author: Jan Stephan
 *
 *      CUDAWeighting manages the concrete implementation of weighting the projections.
 */

#ifndef CUDAWEIGHTING_H_
#define CUDAWEIGHTING_H_

#include "../image/Image.h"
#include "../image/StdImage.h"

#include "CUDAImage.h"

namespace ddafa
{
	namespace impl
	{
		class CUDAWeighting
		{
			public:
				using input_image_type = ddafa::image::Image<float, StdImage<float>>;
				using output_image_type = ddafa::image::Image<float, CUDAImage<float>>;

			public:
				CUDAWeighting();
				void process(input_image_type&& img);
				output_image_type wait();

			protected:
				~CUDAWeighting();
		};
	}
}


#endif /* CUDAWEIGHTING_H_ */
