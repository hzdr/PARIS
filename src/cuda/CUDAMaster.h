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

			protected:
				~CUDAMaster();

			private:
				int device_;
		};
	}
}



#endif /* CUDAMASTER_H_ */
