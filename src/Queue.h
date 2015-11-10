/*
 * Queue.h
 *
 *  Created on: 10.11.2015
 *      Author: Jan Stephan
 *
 *      A Queue object is between two pipeline stages and handles the data that is passed from one
 *      to another. It is thread-safe.
 */

#ifndef QUEUE_H_
#define QUEUE_H_

#include <mutex>
#include <queue>

template <class Object>
class Queue
{
	public:

		Queue()
		: objects_(), mutex_()
		{
		}

		/*
		 * Appends an object to the end of the Queue
		 */
		void push(Object object)
		{
			std::lock_guard<std::mutex> lock(mutex_);
			objects_.push(object);
		}

		/*
		 * Returns the next object in the queue and deletes it internally. Always
		 * check for emptiness with empty() before using this.
		 */
		Object take() noexcept
		{
			std::lock_guard<std::mutex> lock(mutex_);
			auto ret = objects_.front();
			objects_.pop();
			return ret;
		}

		bool empty() const noexcept
		{
			std::lock_guard<std::mutex> lock(mutex_);
			return objects_.empty();
		}

	private:
		std::queue<Object> objects_;
		mutable std::mutex mutex_;
};


#endif /* QUEUE_H_ */
