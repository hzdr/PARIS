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
			protected:
				using task_type = ddafa::master_worker::Task<CUDATask>;
				using result_type = task_type;

				auto start() -> void;
				auto process(task_type&& current_task) -> result_type;

				~CUDAWorker() = default;
		};
	}
}



#endif /* CUDAWORKER_H_ */
