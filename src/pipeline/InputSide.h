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
				void input(InputType&& input)
				{
					input_queue_.push(std::forward<InputType&&>(input));
				}

			protected:
				Queue<InputType> input_queue_;
		};
	}
}


#endif /* INPUTSIDE_H_ */
