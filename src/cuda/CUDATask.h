/*
 * CUDATask.h
 *
 *  Created on: 12.11.2015
 *      Author: Jan Stephan
 *
 *      CUDA implementation policy for the Task class.
 */

#ifndef CUDATASK_H_
#define CUDATASK_H_

namespace ddafa
{
	namespace impl
	{
		class CUDATask
		{
			public:
				using data_type = float;
				using result_type = float;

			protected:
				~CUDATask();
		};
	}
}



#endif /* CUDATASK_H_ */
