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

#include <memory>
#include <utility>
#include <vector>

#include "Task.h"
#include "../common/Queue.h"
#include "Worker.h"

namespace ddafa
{
	namespace master_worker
	{
		template <class Implementation, class... Args>
		class Master : public Implementation
		{
			public:
				Master(Args&&... args)
				: Implementation(std::forward<Args&&>(args)...)
				, task_queue_{std::make_shared<ddafa::common::Queue<Task<typename Implementation::task_type>>>()}
				, result_queue_ {std::make_shared<ddafa::common::Queue<Task<typename Implementation::task_type>>>()}
				{
				}

				/*
				 * Move constructor
				 */
				Master(Master&& other) noexcept
				: Implementation(std::forward<Master&&>(other))
				, workers_{std::move(other.workers_)}
				, task_queue_{std::move(other.task_queue_)}
				, result_queue_{std::move(other.result_queue_)}
				{
				}

				/*
				 * Move operator
				 */
				Master& operator=(Master&& rhs) noexcept
				{
					workers_ = std::move(rhs.workers_);
					task_queue_ = std::move(rhs.task_queue_);
					result_queue_ = std::move(rhs.result_queue_);

					return *this;
				}


			private:
				/*
				 * Disable copies
				 */
				Master(const Master& other) = delete;
				Master& operator=(const Master& rhs) = delete;

			private:
				std::vector<Worker<typename Implementation::worker_type>> workers_;
				std::shared_ptr<ddafa::common::Queue<Task<typename Implementation::task_type>>> task_queue_;
				std::shared_ptr<ddafa::common::Queue<Task<typename Implementation::task_type>>> result_queue_;
		};
	}
}



#endif /* MASTER_H_ */
