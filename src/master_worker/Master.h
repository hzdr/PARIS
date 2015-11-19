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

#include <functional>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

#include "Task.h"
#include "../common/Queue.h"
#include "../image/Image.h"
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
					for(auto i = 0; i < Implementation::workerCount(); ++i)
						workers_.emplace_back(task_queue_, result_queue_);
				}

				/*
				 * Move constructor
				 */
				Master(Master&& other) noexcept
				: Implementation(std::forward<Master&&>(other))
				, workers_{std::move(other.workers_)}
				, worker_threads_{std::move(other.worker_threads_)}
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
					worker_threads_ = std::move(rhs.worker_threads_);
					task_queue_ = std::move(rhs.task_queue_);
					result_queue_ = std::move(rhs.result_queue_);

					return *this;
				}

				void start()
				{
					Implementation::start();
					for(auto&& worker : workers_)
						worker_threads_.emplace_back(&Worker<typename Implementation::worker_type>::start,
														&worker);

					while(true)
					{
						const ddafa::image::Image* img_ptr = input_queue_.take();
						if(!img_ptr->valid())
							break; // poisonous pill

						task_queue_->push(Implementation::createTask(img_ptr));
					}

					task_queue_->push(Implementation::createTask(nullptr)); // send poisonous pill to workers
					stop();
				}

				void stop()
				{
					Implementation::stop();
					for(auto&& thread : worker_threads_)
						thread.join();
				}

				void input(const ddafa::image::Image* img)
				{
					input_queue_.push(img);
				}

			private:
				/*
				 * Disable copies
				 */
				Master(const Master& other) = delete;
				Master& operator=(const Master& rhs) = delete;

			private:
				std::vector<Worker<typename Implementation::worker_type>> workers_;
				std::vector<std::thread> worker_threads_;
				ddafa::common::Queue<const ddafa::image::Image*> input_queue_;
				std::shared_ptr<ddafa::common::Queue<Task<typename Implementation::task_type>>> task_queue_;
				std::shared_ptr<ddafa::common::Queue<Task<typename Implementation::task_type>>> result_queue_;
		};
	}
}



#endif /* MASTER_H_ */
