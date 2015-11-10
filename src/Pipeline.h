/*
 * Pipeline.h
 *
 *  Created on: 10.11.2015
 *      Author: Jan Stephan
 *
 *      The Pipeline object manages the connection of the individual stages.
 */

#ifndef PIPELINE_H_
#define PIPELINE_H_

#include "InputSide.h"
#include "OutputSide.h"

class Pipeline
{
	public:

	/*
	 * Connects two stages. Note that only two stages can be connected to each other on each side.
	 */
	template <class Data>
	void connect(OutputSide<Data>& first, InputSide<Data>& second)
	{
		first.attach(second);
	}
};


#endif /* PIPELINE_H_ */
