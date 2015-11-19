/*
 * CUDAWorker.cu
 *
 *  Created on: 12.11.2015
 *      Author: Jan Stephan
 *
 *      CUDA implementation policy for the Worker class. Implementation file.
 */

#include <iostream>

#include "CUDAWorker.h"

#include "../master_worker/Task.h"

namespace ddafa
{
	namespace impl
	{
		CUDAWorker::~CUDAWorker()
		{
		}

		void CUDAWorker::start()
		{
		}

		ddafa::master_worker::Task<CUDAWorker::task_type>
		CUDAWorker::process(ddafa::master_worker::Task<task_type>&& current_task)
		{
			std::cout << "CUDAWorker: STUB: process() called" << std::endl;
			return ddafa::master_worker::Task<CUDAWorker::task_type>(0, nullptr, nullptr);
		}
	}
}
