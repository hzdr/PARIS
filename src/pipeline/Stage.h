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
		: public InputSide<ddafa::image::Image>
		, public OutputSide<ddafa::image::Image>
		, public Implementation
		{
			public:
				Stage(Args&&... args)
				: InputSide<ddafa::image::Image>()
				, OutputSide<ddafa::image::Image>()
				, Implementation(std::forward<Args&&>(args)...)
				{
				}

				void run()
				{
					while(true)
					{
						ddafa::image::Image img = input_queue_.take();
						if(img.valid())
							Implementation::process(std::move(img));
						else
						{
							// received poisonous pill, fetch results and end
							Implementation::process(std::move(img));
							auto result = Implementation::wait();
							output(std::move(result));
							break;
						}
					}
				}

		};
	}
}


#endif /* STAGE_H_ */
