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
#include <type_traits>
#include <utility>

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
				ddafa::image::Image<T, StdImage<T>> loadImage(std::string path)
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
					auto image_header = std::unique_ptr<std::uint8_t>(new std::uint8_t[header.image_header_size]);
					readEntry(file, image_header.get(), header.image_header_size);
						// ...
					image_header.reset();

					// calculate dimensions
					std::uint32_t width = header.brx - header.ulx + 1;
					std::uint32_t height = header.bry - header.uly + 1;
					std::uint32_t number_of_projections  = header.number_of_frames;
					if(number_of_projections > 1)
						throw std::runtime_error("HIS loader: No support for more than one projection per file");

					// read image data
					std::unique_ptr<T, typename StdImage<T>::deleter_type> img_buffer(new T[width * height]);

					switch(header.type_of_numbers)
					{
						case tn_unsigned_char:
						{
							std::unique_ptr<std::uint8_t> buffer(new std::uint8_t[width * height]);
							readEntry(file, buffer.get(), width * height * sizeof(std::uint8_t));
							readBuffer<T, std::uint8_t>(img_buffer.get(), buffer.get(), width, height);
							break;
						}

						case tn_unsigned_short:
						{
							std::unique_ptr<std::uint16_t> buffer(new std::uint16_t[width * height]);
							readEntry(file, buffer.get(), width * height * sizeof(std::uint16_t));
							readBuffer<T, std::uint16_t>(img_buffer.get(), buffer.get(), width, height);
							break;
						}

						case tn_dword:
						{
							std::unique_ptr<std::uint32_t> buffer(new std::uint32_t[width * height]);
							readEntry(file, buffer.get(), width * height * sizeof(std::uint32_t));
							readBuffer<T, std::uint32_t>(img_buffer.get(), buffer.get(), width, height);
							break;
						}

						case tn_double:
						{
							std::unique_ptr<double> buffer(new double[width * height]);
							readEntry(file, buffer.get(), width * height * sizeof(double));
							readBuffer<T, double>(img_buffer.get(), buffer.get(), width, height);
							break;
						}

						case tn_float:
						{
							std::unique_ptr<float> buffer(new float[width * height]);
							readEntry(file, buffer.get(), width * height * sizeof(float));
							readBuffer<T, float>(img_buffer.get(), buffer.get(), width, height);
							break;
						}

						default:
							throw std::runtime_error("HIS loader: No implementation for data type of file "
														+ path);
					}

					return ddafa::image::Image<T, StdImage<T>>(width, height, std::move(img_buffer));
				}

			protected:
				~HIS() {}

			private:
				template <typename T>
				inline void readEntry(std::ifstream& file, T& entry)
				{
					file.read(reinterpret_cast<char *>(&entry), sizeof(entry));
				}

				template <typename T>
				inline void readEntry(std::ifstream& file, T* entry, std::size_t size)
				{
					file.read(reinterpret_cast<char *>(entry), size);
				}

				template <typename Wanted, typename Actual>
				inline typename std::enable_if<std::is_same<Wanted, Actual>::value>::type
				readBuffer(Wanted* dest, Actual* buf, std::uint32_t width, std::uint32_t height)
				{
					for(std::size_t j = 0; j < height; ++j)
					{
						for(std::size_t i = 0; i < width; ++i)
							dest[i + j * width] = buf[i + j * width];
					}
				}

				template <typename Wanted, typename Actual>
				inline typename std::enable_if<!std::is_same<Wanted, Actual>::value>::type
				readBuffer(Wanted* dest, Actual* buf, std::uint32_t width, std::uint32_t height)
				{
					for(std::size_t j = 0; j < height; ++j)
					{
						for(std::size_t i = 0; i < width; ++i)
							dest[i + j * width] = Wanted(buf[i + j * width]);
					}
				}
		};
	}
}


#endif /* HIS_H_ */
