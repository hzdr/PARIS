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

#include <string>

#include "Image.h"

template <class HandlerPolicy>
class ImageHandler : public HandlerPolicy
{
	public:
		Image loadImage(std::string path)
		{
			return HandlerPolicy::loadImage(path);
		}

		void saveImage(Image image, std::string path)
		{
				HandlerPolicy::saveImage(image, path);
		}
};


#endif /* IMAGEHANDLER_H_ */
