/*
 * ImageHandler.h
 *
 *  Created on: 05.11.2015
 *      Author: Jan Stephan
 *
 *      ImageHandler class that loads and saves images based on the specific handler policies.
 */

#ifndef IMAGEHANDLER_H_
#define IMAGEHANDLER_H_

#include <cstddef>
#include <string>
#include <utility>

#include "Image.h"

namespace ddafa
{
	namespace image
	{
		template <class Implementation>
		class ImageHandler : public Implementation
		{
			public:
				/*
				 * Loads an image from the given path. The image data will be converted to float if needed.
				 */
				Image loadImage(std::string path)
				{
					return Implementation::loadImage(path);
				}

				/*
				 * Saves an image to the given path. The image will be saved in floating point format.
				 */
				void saveImage(Image&& image, std::string path)
				{
					Implementation::saveImage(std::forward<Image&&>(image), path);
				}

				/*
				 * Saves an image into a volume at the given path. The volume will be saved in floating
				 * point format.
				 */
				void saveToVolume(Image&& image, std::string path, std::size_t index)
				{
					Implementation::saveToVolume(std::forward<Image&&>(image), path, index);
				}
		};
	}
}



#endif /* IMAGEHANDLER_H_ */
