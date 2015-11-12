/*
 * Master.h
 *
 *  Created on: 12.11.2015
 *      Author: Jan Stephan
 *
 *      The Master object distributes the work among its Worker objects. The concrete implementation is
 *      done in the Implementation policy.
 */

#ifndef MASTER_H_
#define MASTER_H_

#include <vector>

#include "Task.h"

template <class Implementation>
class Master : public Implementation
{
	public:
		Master()
		: Implementation()
		{
		}

	private:
		/*
		 * Disable copies
		 */
		Master(const Master& other) = delete;
		Master& operator=(const Master& rhs) = delete;

	private:
		std::vector<decltype(Implementation::workerImplementationType())> workers_;
};


#endif /* MASTER_H_ */
