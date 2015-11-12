/*
 * Queue.h
 *
 *  Created on: 12.11.2015
 *      Author: Jan Stephan
 *
 *      Thread-safe queue for sharing objects.
 */

#ifndef QUEUE_H_
#define QUEUE_H_

#include <condition_variable>
#include <mutex>
#include <queue>

namespace ddafa
{
	namespace common
	{
		template <class Object>
		class Queue
		{
			public:
				void push(Object&& object)
				{
					std::unique_lock<std::mutex> lock(mutex_);
					queue_.push(object);
					cv_.notify_one();
				}

				Object take()
				{
					std::unique_lock<std::mutex> lock(mutex_);
					while(queue_.empty())
						cv_.wait(lock);

					Object ret = queue_.front();
					queue_.pop();
					return ret;
				}

			private:
				std::mutex mutex_;
				std::condition_variable cv_;
				std::queue<Object> queue_;

		};
	}
}




#endif /* QUEUE_H_ */
