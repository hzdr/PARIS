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

#include "../common/Geometry.h"
#include "../common/Queue.h"
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
				using input_type = ddafa::image::Image<float, StdImage<float>>;
				using output_type = ddafa::image::Image<float, CUDAImage<float>>;

			public:
				CUDAWeighting(ddafa::common::Geometry geo);
				void process(input_type&& img);
				output_type wait();

			protected:
				~CUDAWeighting();

			private:
				ddafa::common::Geometry geo_;
				ddafa::common::Queue<output_type> results_;
				float h_min_;
				float v_min_;
		};
	}
}


#endif /* CUDAWEIGHTING_H_ */
