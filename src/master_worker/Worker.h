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

#include <memory>
#include <utility>

#include "../common/Queue.h"
#include "Task.h"

namespace ddafa
{
	namespace master_worker
	{
		template <class Implementation>
		class Worker : public Implementation
		{
			public:
				/*
				 * Constructs a new worker.
				 */
				Worker( std::weak_ptr<ddafa::common::Queue<Task<typename Implementation::task_type>>> task_queue,
						std::weak_ptr<ddafa::common::Queue<Task<typename Implementation::task_type>>> result_queue)
				: Implementation(), task_queue_{task_queue.lock()}, result_queue_{result_queue.lock()}
				{
				}

				/*
				 * Move constructor
				 */
				Worker(Worker&& other) noexcept
				: Implementation()
				, task_queue_{std::move(other.task_queue_)}
				, result_queue_{std::move(other.result_queue_)}
				{
				}

				/*
				 * Move operator
				 */
				Worker& operator=(Worker&& rhs)
				{
					task_queue_ = std::move(rhs.task_queue_);
					result_queue_ = std::move(rhs.result_queue_);
					return *this;
				}

				void start()
				{
					Implementation::start();

					while(true)
					{
						auto&& task = task_queue_->take();
						if(!task.valid())
							break;

						result_queue_->push(Implementation::process(std::move(task)));
					}
				}

			private:
				/*
				 * Disable copies
				 */
				Worker(const Worker& other) = delete;
				Worker& operator=(const Worker& rhs) = delete;

			private:
				std::shared_ptr<ddafa::common::Queue<Task<typename Implementation::task_type>>> task_queue_;
				std::shared_ptr<ddafa::common::Queue<Task<typename Implementation::task_type>>> result_queue_;
		};
	}
}



#endif /* WORKER_H_ */
