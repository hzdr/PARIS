/*
 * Worker.h
 *
 *  Created on: 12.11.2015
 *      Author: Jan Stephan
 *
 *      The Worker object receives a Task from its corresponding Master object and will forward this Task to
 *      its concrete Implementation policy.
 */

#ifndef WORKER_H_
#define WORKER_H_

#include "Task.h"

template <class Implementation>
class Worker
{
	public:
		Worker()
		: Implementation()
		{
		}
};


#endif /* WORKER_H_ */
