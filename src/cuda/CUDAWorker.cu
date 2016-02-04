/*
 * CUDAWorker.cu
 *
 *  Created on: 12.11.2015
 *      Author: Jan Stephan
 *
 *      CUDA implementation policy for the Worker class. Implementation file.
 */

#define BOOST_ALL_DYN_LINK
#include <boost/log/trivial.hpp>

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
			BOOST_LOG_TRIVIAL(warning) << "CUDAWorker: STUB: process() called";
			return result_type(0, nullptr, nullptr);
		}
	}
}
