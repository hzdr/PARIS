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

template <class HandlerPolicy>
class ImageHandler : public HandlerPolicy
{
	public:
		/*
		 * Loads an image from the given path. The image data will be converted to float if needed.
		 */
		Image loadImage(std::string path)
		{
			return HandlerPolicy::loadImage(path);
		}

		/*
		 * Saves an image to the given path (lvalue version). The image will be saved in floating point format.
		 */
		void saveImage(const Image& image, std::string path)
		{
			HandlerPolicy::saveImage(image, path);
		}

		/*
		 * Saves an image to the given path (rvalue version). The image will be saved in floating point format.
		 */
		void saveImage(Image&& image, std::string path)
		{
			HandlerPolicy::saveImage(std::forward<Image&&>(image), path);
		}

		/*
		 * Saves an image into a volume at the given path (lvalue version). The volume will be saved in floating
		 * point format.
		 */
		void saveToVolume(const Image& image, std::string path, std::size_t index)
		{
			HandlerPolicy::saveToVolume(image, path, index);
		}

		/*
		 * Saves an image into a volume at the given path (rvalue version). The volume will be saved in floating
		 * point format.
		 */
		void saveToVolume(Image&& image, std::string path, std::size_t index)
		{
			HandlerPolicy::saveToVolume(std::forward<Image&&>(image), path, index);
		}
};


#endif /* IMAGEHANDLER_H_ */
