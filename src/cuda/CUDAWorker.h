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

class CUDAWorker
{
	public:
		using task_type = CUDATask;
};


#endif /* CUDAWORKER_H_ */
