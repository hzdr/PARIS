/*
 * Pipeline.h
 *
 *  Created on: 12.11.2015
 *      Author: Jan Stephan
 */

#ifndef PIPELINE_H_
#define PIPELINE_H_

#include <memory>
#include <thread>

#include "InputSide.h"
#include "OutputSide.h"
#include "Port.h"
#include "SourceStage.h"
#include "SinkStage.h"

namespace ddafa
{
	namespace pipeline
	{
		template <class Data>
		void connect(std::shared_ptr<OutputSide<Data>> first, std::shared_ptr<InputSide<Data>> second)
		{
			Port *port = new Port;
			first->attach(port);
			port->attach(second);
		}
	}
}


#endif /* PIPELINE_H_ */
