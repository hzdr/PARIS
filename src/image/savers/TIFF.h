/*
 * TIFF.h
 *
 *  Created on: 05.11.2015
 *      Author: Jan Stephan
 *
 *      An implementation for the ImageHandler class that loads and saves TIFF images and volumes.
 */

#ifndef TIFF_H_
#define TIFF_H_

#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include <tiffio.h>

#include "../Image.h"
#include "../StdImage.h"

namespace ddafa
{
	namespace impl
	{
		template <typename T, class Allocator = std::allocator<T>, class Deleter = std::default_delete<T>>
		class TIFF : public Allocator
		{
			public:
				using allocator_type = Allocator;
				using deleter_type = Deleter;
				using image_type = ddafa::impl::StdImage<T, Allocator, Deleter>;

			public:
					template <typename U>
					typename std::enable_if<std::is_same<T, U>::value, void>::type
					saveImage(ddafa::image::Image<U, image_type>&& image, std::string& path)
					{
						path.append(".tif");
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

					//TODO: saveToVolume

			protected:
					~TIFF() = default; // disable undefined behavior
		};
	}
}

#endif /* TIFF_H_ */
