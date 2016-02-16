/*
 * InputSide.h
 *
 *  Created on: 12.11.2015
 *      Author: Jan Stephan
 *
 *      The InputSide specifies the input interface of a pipeline stage.
 */

#ifndef INPUTSIDE_H_
#define INPUTSIDE_H_

#include <utility>

#include "../common/Queue.h"

namespace ddafa
{
	namespace pipeline
	{
		template <class InputType>
		class InputSide
		{
			public:
				auto input(InputType&& in) -> void
				{
					input_queue_.push(std::forward<InputType&&>(in));
				}

			protected:
				ddafa::common::Queue<InputType> input_queue_;
		};
	}
}


#endif /* INPUTSIDE_H_ */
