/*
 * OutputSide.h
 *
 *  Created on: 10.11.2015
 *      Author: Jan Stephan
 *
 *      OutputSide defines a Stage's interface for the output of processed data.
 */

#ifndef OUTPUTSIDE_H_
#define OUTPUTSIDE_H_

#include <utility>

#include "InputSide.h"

template <class OutputData>
class OutputSide
{
	public:
		void attach(InputSide<OutputData>& next_stage)
		{
			next_ = next_stage;
		}

		void output(OutputData&& output_data)
		{
			next_.input(std::forward<OutputData&&>(output_data));
		}

	private:
		InputSide<OutputData>& next_;
};


#endif /* OUTPUTSIDE_H_ */
