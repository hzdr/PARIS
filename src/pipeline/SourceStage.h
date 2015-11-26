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

#ifdef DDAFA_DEBUG
#include <iostream>
#endif
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
		class SourceStage : public OutputSide<typename ImageHandler::image_type>, public ImageHandler
		{
			public:
				using output_type = typename ImageHandler::image_type;

			public:
				SourceStage(std::string path)
				: OutputSide<output_type>(), ImageHandler(), dir_string_{path}
				{
				}

				void run()
				{
					// TODO: read target directory
					output_type img = ImageHandler::loadImage("my/fancy/path.tif");
					if(img.valid())
						this->output(std::move(img));
					else
						throw std::runtime_error("Invalid image file: %PATH");

					// all images loaded, send poisonous pill
#ifdef DDAFA_DEBUG
					std::cout << "SourceStage: Loading complete, sending poisonous pill" << std::endl;
#endif
					this->output(output_type());
				}

			private:
				std::string dir_string_;
		};
	}
}


#endif /* SOURCESTAGE_H_ */
