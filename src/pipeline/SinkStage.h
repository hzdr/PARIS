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

#ifdef DDAFA_DEBUG
#include <iostream>
#endif

#include <string>
#include <utility>

#include "../image/Image.h"

#include "InputSide.h"

namespace ddafa
{
	namespace pipeline
	{
		template <class ImageSaver>
		class SinkStage : public ImageSaver
						, public InputSide<ddafa::image::Image<float, typename ImageSaver::image_type>>
		{
			public:
				using input_type = ddafa::image::Image<float, typename ImageSaver::image_type>;

			public:
				SinkStage(const std::string& path, const std::string& prefix)
				: InputSide<input_type>(), ImageSaver(), target_dir_{path}, prefix_{prefix}
				{
				}

				void run()
				{
					while(true)
					{
						input_type img = this->input_queue_.take();
						if(img.valid())
							ImageSaver::template saveImage<float>(std::move(img), "/media/HDD1/Feldkamp/out.tif");
						else
						{
#ifdef DDAFA_DEBUG
							std::cout << "SinkStage: Poisonous pill arrived, terminating." << std::endl;
#endif
							break; // poisonous pill
						}

					}
				}

			private:
				std::string target_dir_;
				std::string prefix_;
		};
	}
}


#endif /* SINKSTAGE_H_ */
