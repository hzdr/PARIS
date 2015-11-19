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
			public:
				using task_type = CUDATask;
				using worker_type = CUDAWorker;

			public:
				CUDAMaster(int device_num);
				CUDAMaster(CUDAMaster&& other);
				void start();
				void stop();

				ddafa::master_worker::Task<task_type> createTask(const ddafa::image::Image* img_ptr);

			protected:
				~CUDAMaster();
				int workerCount() const noexcept;

			private:
				int device_;
				int number_of_workers_;
		};
	}
}



#endif /* CUDAMASTER_H_ */
