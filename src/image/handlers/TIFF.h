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

namespace ddafa
{
	namespace impl
	{
		class TIFF
		{
		public:
			ddafa::image::Image loadImage(std::string path);
			void saveImage(ddafa::image::Image&& image, std::string path);
			//TODO: saveToVolume

		protected:
			~TIFF(); // disable undefined behavior
		};
	}
}

#endif /* TIFFHANDLER_H_ */
