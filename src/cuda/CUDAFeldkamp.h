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

#include "CUDAHostAllocator.h"
#include "CUDAHostDeleter.h"
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
				using input_type = ddafa::image::Image<float, CUDAImage<float>>;
				using output_type = ddafa::image::Image<float, StdImage<float, CUDAHostAllocator<float>, CUDAHostDeleter>>;

			public:
				CUDAFeldkamp();
				auto process(input_type&&) -> void;
				auto wait() -> output_type;

			protected:
				~CUDAFeldkamp();

			private:
				std::vector<master_type> masters_;
				std::vector<std::thread> master_threads_;
		};
	}
}


#endif /* CUDAFELDKAMP_H_ */
