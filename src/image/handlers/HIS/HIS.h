/*
 * HIS.h
 *
 *  Created on: 26.11.2015
 *      Author: Jan Stephan
 *
 *      An implementation for the ImageHandler class that loads and saves HIS images and volumes.
 */

#ifndef HIS_H_
#define HIS_H_

#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>

#include "../../Image.h"
#include "../../StdImage.h"

#include "HISHeader.h"

namespace ddafa
{
	namespace impl
	{
		class HIS
		{
			private:
				enum Datatype
				{
					tn_not_implemented = -1,
					tn_unsigned_char	= 2,
					tn_unsigned_short	= 4,
					tn_dword			= 32,
					tn_double			= 64,
					tn_float			= 128
				};

			public:
				template <typename T>
				ddafa::image::Image<T, ddafa::impl::StdImage<T>> loadImage(std::string path)
				{
					// read file header
					HISHeader header;

					std::ifstream file(path.c_str(), std::ios_base::binary);
					if(!file.is_open())
						throw std::runtime_error("HIS loader: Could not open file " + path);

					readEntry(file, header.file_type);
					readEntry(file, header.header_size);
					readEntry(file, header.header_version);
					readEntry(file, header.file_size);
					readEntry(file, header.image_header_size);
					readEntry(file, header.ulx);
					readEntry(file, header.uly);
					readEntry(file, header.brx);
					readEntry(file, header.bry);
					readEntry(file, header.number_of_frames);
					readEntry(file, header.correction);
					readEntry(file, header.integration_time);
					readEntry(file, header.type_of_numbers);
					readEntry(file, header.x);

					if(header.file_type != HIS_FILE_ID)
						throw std::runtime_error("HIS loader: File " + path + " is not a valid HIS file.");

					if(header.header_size != HIS_FILE_HEADER_SIZE)
						throw std::runtime_error("HIS loader: File header size mismatch for file " + path);

					if(header.type_of_numbers == tn_not_implemented)
						throw std::runtime_error("HIS loader: No implementation for datatype of file " + path);

					// jump over image header
					std::uint8_t* image_header = new std::uint8_t[header.image_header_size];
					readEntry(file, image_header, header.image_header_size);
						// ...
					delete[] imageHeader;

					// calculate dimensions
					std::uint32_t width = header.brx - header.ulx + 1;
					std::uint32_t height = header.bry - header.uly + 1;
					std::uint32_t number_of_projections  = header.number_of_frames;

					// read image data
					std::size_t buffer_size = width * height * sizeoftype(header.type_of_numbers);
					std::unique_ptr<std::uint8_t> buffer(new std::uint8_t[buffer_size]);
					std::unique_ptr<T, typename StdImage::deleter_type> img_buffer(new T[width * height]);

					switch(header.type_of_numbers)
					{

					}
				}

			protected:
				~HIS();

			private:
				template <typename T>
				inline void readEntry(std::ifstream& file, T& entry)
				{
					file.read(reinterpret_cast<char *>(&entry), sizeof(entry));
				}

				template <typename T>
				inline explicit void readEntry(std::ifstream& file, T* entry, std::size_t size)
				{
					file.read(reinterpret_cast<char *>(entry), size);
				}

				inline std::size_t sizeoftype(std::uint16_t type)
				{
					switch(type)
					{
						case tn_unsigned_char:	return sizeof(std::uint8_t);
						case tn_unsigned_short:	return sizeof(std::uint16_t);
						case tn_dword:			return sizeof(std::uint32_t);
						case tn_double:			return sizeof(double);
						case tn_float:			return sizeof(float);
						default:				throw std::runtime_error("HIS loader: Invalid data type");
					}
				}
		};
	}
}


#endif /* HIS_H_ */
