/*
 * ImageLoader.h
 *
 *  Created on: 26.01.2016
 *      Author: Jan Stephan
 *
 *      ImageLoad class that loads images based on the specific handler policies.
 */

#ifndef IMAGELOADER_H_
#define IMAGELOADER_H_

#include <memory>
#include <string>

#include "Image.h"

namespace ddafa
{
	namespace image
	{
		template <class Implementation>
		class ImageLoader : public Implementation
		{
			public:
				using image_type = typename Implementation::image_type;

			public:
				/*
				 * Loads an image from the given path. The image data will be converted to the given
				 * data type if needed.
				 */
				template <typename T>
				auto loadImage(const std::string& path) -> decltype(Implementation::template loadImage<T>(path))
				{
					return Implementation::template loadImage<T>(path);
				}
		};
	}
}



#endif /* IMAGELOADER_H_ */
