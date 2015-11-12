/*
 * SourceStage.h
 *
 *  Created on: 12.11.2015
 *      Author: Jan Stephan
 *
 *      The SourceStage uses its ImageHandler policy to load images from the specified path. It will then
 *      forward those images to the next pipeline stage.
 */

#ifndef SOURCESTAGE_H_
#define SOURCESTAGE_H_

#include <stdexcept>
#include <string>
#include <utility>

#include "../image/Image.h"

#include "OutputSide.h"

namespace ddafa
{
	namespace pipeline
	{
		template <class ImageHandler>
		class SourceStage : public OutputSide<ddafa::image::Image>, public ImageHandler
		{
			public:
				SourceStage(std::string path)
				: OutputSide<ddafa::image::Image>(), ImageHandler(), dir_string_{path}
				{
				}

				void start()
				{
					// TODO: read target directory
					ddafa::image::Image img = ImageHandler::loadImage("my/fancy/path.tif");
					if(img.valid())
						output(std::move(img));
					else
						throw std::runtime_error("Invalid image file: %PATH");

					// all images loaded, send poisonous pill
					output(ddafa::image::Image());
				}

			private:
				std::string dir_string_;
		};
	}
}


#endif /* SOURCESTAGE_H_ */
