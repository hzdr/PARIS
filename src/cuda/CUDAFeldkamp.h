/*
 * CUDAFeldkamp.h
 *
 *  Created on: 12.11.2015
 *      Author: Jan Stephan
 *
 *      This class is the concrete backprojection implementation for the Stage class.
 */

#ifndef CUDAFELDKAMP_H_
#define CUDAFELDKAMP_H_

#include <thread>
#include <vector>

#include "../image/Image.h"
#include "../image/StdImage.h"

#include "../master_worker/Master.h"

#include "CUDAImage.h"
#include "CUDAMaster.h"

namespace ddafa
{
	namespace impl
	{
		class CUDAFeldkamp
		{
			private:
				using master_type = ddafa::master_worker::Master<CUDAMaster, int&>;
				using input_image_type = ddafa::image::Image<float, CUDAImage<float>>;
				using output_image_type = ddafa::image::Image<float, StdImage<float>>;

			public:
				CUDAFeldkamp();
				void process(input_image_type&& img);
				output_image_type wait();

			protected:
				~CUDAFeldkamp();

			private:
				std::vector<master_type> masters_;
				std::vector<std::thread> master_threads_;
		};
	}
}


#endif /* CUDAFELDKAMP_H_ */
