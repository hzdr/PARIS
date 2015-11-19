/*
 * TIFF.cpp
 *
 *  Created on: 05.11.2015
 *      Author: Jan Stephan
 *
 *      Implementation file for the TIFF ImageHandler implementation.
 */

#include <cstdint>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>

#include <tiffio.h>

#include "../Image.h"
#include "../StdImage.h"
#include "TIFF.h"

namespace ddafa
{
	namespace impl
	{
		TIFF::image_type TIFF::loadImage(std::string path)
		{
			// as TIFF is an incomplete type we cannot wrap it into unique_ptr
			::TIFF* tif = TIFFOpen(path.c_str(), "r");
			if(tif == nullptr)
			{
				TIFFClose(tif);
				return image_type(); // return invalid image
			}

			// everything okay, time to read
			// note that these are not the standard integer types - they will hopefully be convertible
			uint32 width, height;

			TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
			TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);

			auto data = new float[width * height];
			auto dataPtr = data;

			// TODO: type checking

			for(uint32 row = 0; row < height; ++row)
			{
				TIFFReadScanline(tif, dataPtr, row);
				dataPtr += width;
			}

			TIFFClose(tif);
			return image_type(width, height, data); // the data is now owned by the Image object
		}

		void TIFF::saveImage(TIFF::image_type&& image, std::string path)
		{
			::TIFF* tif = TIFFOpen(path.c_str(), "w");
			if(tif == nullptr)
			{
				TIFFClose(tif);
				throw std::runtime_error("TIFFHandler: Could not open file " + path + " for writing.");
			}

			TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, image.width());
			TIFFSetField(tif, TIFFTAG_IMAGELENGTH, image.height());
			TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);
			TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 32);
			TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);

			auto data = image.data();
			auto dataPtr = data;

			for(uint32 row = 0; row < image.height(); ++row)
			{
				TIFFWriteScanline(tif, dataPtr, row);
				dataPtr += image.width();
			}

			TIFFClose(tif);
		}

		// TODO: saveVolume

		TIFF::~TIFF()
		{
		}
	}
}
