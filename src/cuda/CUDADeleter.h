/*
 * CUDADeleter.h
 *
 *  Created on: 19.11.2015
 *      Author: Jan Stephan
 *
 *      A custom deleter for CUDA memory that is managed by smart pointers.
 */

#ifndef CUDADELETER_H_
#define CUDADELETER_H_

namespace ddafa
{
	namespace impl
	{
		struct CUDADeleter
		{
			void operator()(void *p);
		};
	}
}

#endif /* CUDADELETER_H_ */
