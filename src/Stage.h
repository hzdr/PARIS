/*
 * Stage.h
 *
 *  Created on: 10.11.2015
 *      Author: Jan Stephan
 *
 *      One or multiple Stages are between the SourceStage and the SinkStage. They get the data
 *      from their InputSides, process it (this is done in the ImplementationPolicy) and then
 *      forward the result to their OutputSide.
 */

#ifndef STAGE_H_
#define STAGE_H_

template <class InputData, class OutputData, class ImplementationPolicy>
class Stage : public InputData, OutputData, ImplementationPolicy
{

};

#endif /* STAGE_H_ */
