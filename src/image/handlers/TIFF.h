/*
 * TIFFHandler.h
 *
 *  Created on: 05.11.2015
 *      Author: Jan Stephan
 *
 *      An implementation for the ImageHandler class that loads and saves TIFF images and volumes.
 */

#ifndef TIFFHANDLER_H_
#define TIFFHANDLER_H_

#include <string>

#include "../Image.h"
#include "../StdImage.h"

namespace ddafa
{
	namespace impl
	{
		class TIFF
		{
			public:
				using image_type = ddafa::image::Image<float, ddafa::impl::StdImage<float>>;

			public:
					image_type loadImage(std::string path);
					void saveImage(image_type&& image, std::string path);
					//TODO: saveToVolume

			protected:
					~TIFF(); // disable undefined behavior
		};
	}
}

#endif /* TIFFHANDLER_H_ */
