/*
 * InputSide.h
 *
 *  Created on: 10.11.2015
 *      Author: Jan Stephan
 *
 *      InputSide defines a Stage's interface for the input of data.
 */

#ifndef INPUTSIDE_H_
#define INPUTSIDE_H_

#include "Queue.h"

template <class InputData>
class InputSide
{
	public:
		void input(InputData input_data)
		{
			queue_.push(input_data);
		}

	protected:
		Queue<InputData> queue_;
};


#endif /* INPUTSIDE_H_ */
