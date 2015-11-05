/*
 * TIFFHandler.h
 *
 *  Created on: 05.11.2015
 *      Author: Jan Stephan
 *
 *      A policy for the ImageHandler class that loads and saves TIFF images and volumes.
 */

#ifndef TIFFHANDLER_H_
#define TIFFHANDLER_H_

#include <string>

#include "../Image.h"

class TIFFHandler
{
	public:
		Image loadImage(std::string path);
		void saveImage(const Image& image, std::string path);
		void saveImage(Image&& image, std::string path);

	protected:
		~TIFFHandler();
};


#endif /* TIFFHANDLER_H_ */
