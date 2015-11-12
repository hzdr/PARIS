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

#include "../master_worker/Master.h"

#include "CUDAMaster.h"

namespace ddafa
{
	namespace impl
	{
		class CUDAFeldkamp
		{
			public:
				CUDAFeldkamp();
				void process(ddafa::image::Image&& img);
				ddafa::image::Image wait();

			protected:
				~CUDAFeldkamp();

			private:
				std::vector<ddafa::master_worker::Master<CUDAMaster, int&>> masters_;
				std::vector<std::thread> master_threads_;
		};
	}
}


#endif /* CUDAFELDKAMP_H_ */
