/*
 * SinkStage.h
 *
 *  Created on: 12.11.2015
 *      Author: Jan Stephan
 *
 *      The SinkStage uses its ImageHandler policy to save images or volumes that are being passed from
 *      previous pipeline stages.
 */

#ifndef SINKSTAGE_H_
#define SINKSTAGE_H_

#include <string>

#include "../image/Image.h"

#include "InputSide.h"

namespace ddafa
{
	namespace pipeline
	{
		template <class ImageHandler>
		class SinkStage : public InputSide<ddafa::image::Image>, public ImageHandler
		{
			public:
				SinkStage(std::string path)
				: InputSide<ddafa::image::Image>(), ImageHandler(), target_dir_{path}
				{
				}

				void wait()
				{
					while(true)
					{
						ddafa::image::Image img = input_queue_.take();
						if(img.valid())
							ImageHandler::saveImage("my/fancy/path.tif");
						else
							break; // poisonous pill
					}
				}

			private:
				std::string target_dir_;
		};
	}
}


#endif /* SINKSTAGE_H_ */
