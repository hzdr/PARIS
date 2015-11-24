/*
 * Pipeline.h
 *
 *  Created on: 12.11.2015
 *      Author: Jan Stephan
 */

#ifndef PIPELINE_H_
#define PIPELINE_H_

#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "Port.h"

namespace ddafa
{
	namespace pipeline
	{
		class Pipeline
		{
			public:
				template <class First, class Second>
				void connect(First first, Second second)
				{
					// this pointer will be managed by "first", don't delete
					auto port = new Port<typename First::element_type::output_type>;
					first->attach(port);
					port->attach(second);
				}

				template <class PipelineStage, typename... Args>
				std::shared_ptr<PipelineStage> create(Args&&... args)
				{
					return std::make_shared<PipelineStage>(std::forward<Args&&>(args)...);
				}

				template <class Stage>
				void run(Stage stage)
				{
					stage_threads_.emplace_back(&Stage::element_type::run, stage);
				}

				template <class Stage, class... Stages>
				void run(Stage stage, Stages... stages)
				{
					run(stage);
					run(stages...);
				}

				void wait()
				{
					for(auto&& t : stage_threads_)
						t.join();
				}

			private:
				std::vector<std::thread> stage_threads_;
		};

	}
}


#endif /* PIPELINE_H_ */
