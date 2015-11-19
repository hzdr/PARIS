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

		CUDAWorker::result_type	CUDAWorker::process(CUDAWorker::task_type&& current_task)
		{
			std::cout << "CUDAWorker: STUB: process() called" << std::endl;
			return result_type(0, nullptr, nullptr);
		}
	}
}
