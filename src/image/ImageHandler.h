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
#include "StdImage.h"

namespace ddafa
{
	namespace image
	{
		template <class Implementation>
		class ImageHandler : public Implementation
		{
			public:
				/*
				 * Loads an image from the given path. The image data will be converted to the given
				 * data type if needed.
				 */
				template <typename Data = float, class ImageImplementation = ddafa::impl::StdImage<Data>>
				Image<Data, ImageImplementation> loadImage(std::string path)
				{
					return Implementation::loadImage(path);
				}

				/*
				 * Saves an image to the given path. The image will be saved in floating point format.
				 */
				template <typename Data = float, class ImageImplementation = ddafa::impl::StdImage<Data>>
				void saveImage(Image<Data, ImageImplementation>&& image, std::string path)
				{
					Implementation::saveImage(std::forward<Image<Data, ImageImplementation>&&>(image), path);
				}

				/*
				 * Saves an image into a volume at the given path. The volume will be saved in floating
				 * point format.
				 */
				template <typename Data = float, class ImageImplementation = ddafa::impl::StdImage<Data>>
				void saveToVolume(Image<Data, ImageImplementation>&& image, std::string path, std::size_t index)
				{
					Implementation::saveToVolume(std::forward<Image<Data, ImageImplementation>&&>(image),
													path, index);
				}
		};
	}
}



#endif /* IMAGEHANDLER_H_ */
