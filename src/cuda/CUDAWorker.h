/*
 * CUDAWorker.h
 *
 *  Created on: 12.11.2015
 *      Author: Jan Stephan
 *
 *      CUDA implementation policy for the Worker class.
 */

#ifndef CUDAWORKER_H_
#define CUDAWORKER_H_

#include "CUDATask.h"

#include "../master_worker/Task.h"

namespace ddafa
{
	namespace impl
	{
		class CUDAWorker
		{
			public:
				using task_type = CUDATask;

			public:
				void start();

				ddafa::master_worker::Task<task_type>
				process(ddafa::master_worker::Task<task_type>&& current_task);

			protected:
				~CUDAWorker();
		};
	}
}



#endif /* CUDAWORKER_H_ */
