/*
 * Task.h
 *
 *  Created on: 12.11.2015
 *      Author: Jan Stephan
 *
 *      The Task is distributed by the Master object among its Worker objects. It contains the data necessary
 *      to execute the task, an unique identifier and previously allocated memory to store the result. It also
 *      contains the function to be executed on the data via the "Implementation" policy.
 */

#ifndef TASK_H_
#define TASK_H_

#include <cstdint>
#include <memory>
#include <utility>

template <class Implementation>
class Task : public Implementation
{
	public:
		/*
		 * Construct a new task. The pointers will be owned by the Task object from here.
		 */
		Task(std::uint32_t task_id,
				typename Implementation::data_type* task_data,
				typename Implementation::result_type* result_data) noexcept
		: id_{task_id}, data_ptr_{task_data}, result_ptr_{result_data}
		{
		}

		/*
		 * Move constructor
		 */
		Task(Task&& other) noexcept
		: id_{other.id_}, data_ptr_{std::move(other.data_ptr_)}, result_ptr_{std::move(other.result_ptr_)}
		{
		}

		/*
		 * Move operator
		 */
		Task& operator=(Task&& rhs) noexcept
		{
			id_ = rhs.id_;
			data_ptr_ = std::move(rhs.data_ptr_);
			result_ptr_ = std::move(rhs.result_ptr_);

			return *this;
		}

		void execute()
		{
			Implementation::execute(data_ptr_.get(), result_ptr_.get());
		}

	private:
		/*
		 * disallow copies
		 */
		Task(const Task& other) = delete;
		Task& operator=(const Task& rhs) = delete;

	private:
		std::uint32_t id_;
		std::unique_ptr<typename Implementation::data_type> data_ptr_;
		std::unique_ptr<typename Implementation::result_type> result_ptr_;
};


#endif /* TASK_H_ */
