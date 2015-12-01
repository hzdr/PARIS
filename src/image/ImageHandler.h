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
				template <typename T>
				Image<T, ddafa::impl::StdImage<T>> loadImage(std::string path)
				{
					return Implementation::template loadImage<T>(path);
				}

				/*
				 * Saves an image to the given path.
				 */
				template <typename T>
				void saveImage(Image<T, ddafa::impl::StdImage<T>>&& image, std::string path)
				{
					Implementation::saveImage(std::forward<Image<T, ddafa::impl::StdImage<T>>&&>(image), path);
				}

				/*
				 * Saves an image into a volume at the given path.
				 */
				template <typename T>
				void saveToVolume(Image<T, ddafa::impl::StdImage<T>>&& image, std::string path, std::size_t index)
				{
					Implementation::saveToVolume(std::forward<Image<T, ddafa::impl::StdImage<T>>&&>(image),
													path, index);
				}
		};
	}
}



#endif /* IMAGEHANDLER_H_ */
