/*
 * CUDAMaster.h
 *
 *  Created on: 12.11.2015
 *      Author: Jan Stephan
 *
 *      CUDA implementation policy for the Master class.
 */

#ifndef CUDAMASTER_H_
#define CUDAMASTER_H_

#include "CUDAImage.h"
#include "CUDATask.h"
#include "CUDAWorker.h"

#include "../image/Image.h"
#include "../master_worker/Task.h"
#include "../master_worker/Worker.h"

namespace ddafa
{
	namespace impl
	{
		class CUDAMaster
		{
			protected:
				using task_type = ddafa::master_worker::Task<CUDATask>;
				using worker_type = ddafa::master_worker::Worker<CUDAWorker>;
				using data_type = typename task_type::data_type;
				using image_type = ddafa::image::Image<data_type, CUDAImage<data_type>>;

				CUDAMaster(int);
				CUDAMaster(CUDAMaster&&);
				~CUDAMaster();

				auto start() -> void;
				auto stop() -> void;
				auto createTask(const image_type*) -> task_type;
				auto workerCount() const noexcept -> int;

			private:
				int device_;
				int number_of_workers_;
		};
	}
}



#endif /* CUDAMASTER_H_ */
