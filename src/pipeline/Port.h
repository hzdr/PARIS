/*
 * Port.h
 *
 *  Created on: 12.11.2015
 *      Author: Jan Stephan
 *
 *      The Port class connects the OutputSide of a pipeline stage to the InputSide of another stage.
 */

#ifndef PORT_H_
#define PORT_H_

#include <memory>
#include <utility>

#include "InputSide.h"

namespace ddafa
{
	namespace pipeline
	{
		template <class DataType>
		class Port
		{
			public:
				void forward(DataType&& data)
				{
					next_.input(std::forward<DataType&&>(data));
				}

				void attach(std::shared_ptr<InputSide<DataType>> next) noexcept
				{
					next_ = next;
				}

			private:
				InputSide<DataType> next_;
		};
	}
}


#endif /* PORT_H_ */
