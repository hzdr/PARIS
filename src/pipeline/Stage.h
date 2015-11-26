/*
 * Stage.h
 *
 *  Created on: 12.11.2015
 *      Author: Jan Stephan
 *
 *      The Stage class represents the pipeline stages that do actual work. It receives Images from the previous
 *      stage, processes them with its Implementation policy and forwards them to the next stage.
 */

#ifndef STAGE_H_
#define STAGE_H_

#ifdef DDAFA_DEBUG
#include <iostream>
#endif
#include <thread>
#include <utility>

#include "../image/Image.h"

#include "InputSide.h"
#include "OutputSide.h"

namespace ddafa
{
	namespace pipeline
	{
		template <class Implementation, class... Args>
		class Stage
		: public InputSide<typename Implementation::input_type>
		, public OutputSide<typename Implementation::output_type>
		, public Implementation
		{
			public:
				using input_type = typename Implementation::input_type;
				using output_type = typename Implementation::output_type;

			public:
				Stage(Args&&... args)
				: InputSide<input_type>()
				, OutputSide<output_type>()
				, Implementation(std::forward<Args&&>(args)...)
				{
				}

				void run()
				{
					std::thread push_thread(&Stage::push, this);
					std::thread take_thread(&Stage::take, this);

					push_thread.join();
					take_thread.join();
				}

				void push()
				{
					while(true)
					{
						input_type img = this->input_queue_.take();
						if(img.valid())
							Implementation::process(std::move(img));
						else
						{
							// received poisonous pill, time to die
#ifdef DDAFA_DEBUG
							std::cout << "Stage: Poisonous pill arrived at push(), cleaning up" << std::endl;
#endif
							Implementation::process(std::move(img));
							break;
						}
					}
				}

				void take()
				{
					while(true)
					{
						output_type result = Implementation::wait();
						if(result.valid())
							output(std::move(result));
						else
						{
#ifdef DDAFA_DEBUG
							std::cout << "Stage: Poisonous pill arrived at take(), cleaning up" << std::endl;
#endif
							output(std::move(result));
							break;
						}
					}
				}

		};
	}
}


#endif /* STAGE_H_ */
