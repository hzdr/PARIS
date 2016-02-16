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

#define BOOST_ALL_DYN_LINK
#include <boost/log/trivial.hpp>

#include "../Image.h"
#include "../StdImage.h"

#include "HISHeader.h"

namespace ddafa
{
	namespace impl
	{
		template <typename T, class Allocator = std::allocator<T>, class Deleter = std::default_delete<T>>
		class HIS : public Allocator
		{
			public:
				using allocator_type = Allocator;
				using deleter_type = Deleter;
				using image_type = StdImage<T, allocator_type, deleter_type>;

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
				// TODO: Implement support for more than one frame per file
				template <typename U>
				auto loadImage(std::string path)
					-> typename std::enable_if<std::is_same<T, U>::value, ddafa::image::Image<U, image_type>>::type
				{
					using empty_return = ddafa::image::Image<U, image_type>;
					// read file header
					auto header = HISHeader{};

					auto& file = std::ifstream{path.c_str(), std::ios_base::binary};
					if(!file.is_open())
					{
						BOOST_LOG_TRIVIAL(warning) << "HIS loader: Could not open file " + path;
						return empty_return();
					}

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
					{
						BOOST_LOG_TRIVIAL(warning) << "HIS loader: File " << path << " is not a valid HIS file.";
						return empty_return();
					}

					if(header.header_size != HIS_FILE_HEADER_SIZE)
					{
						BOOST_LOG_TRIVIAL(warning) << "HIS loader: File header size mismatch for file " << path;
						return empty_return();
					}

					if(header.type_of_numbers == tn_not_implemented)
					{
						BOOST_LOG_TRIVIAL(warning) << "HIS loader: No implementation for datatype of file " << path;
						return empty_return();
					}

					// jump over image header
					auto image_header = std::unique_ptr<std::uint8_t>(new std::uint8_t[header.image_header_size]);
					readEntry(file, image_header.get(), header.image_header_size);
						// ...
					image_header.reset();

					// calculate dimensions
					auto width = header.brx - header.ulx + 1;
					auto height = header.bry - header.uly + 1;
					auto number_of_projections  = header.number_of_frames;
					if(number_of_projections > 1)
					{
						BOOST_LOG_TRIVIAL(warning) << "HIS loader: No support for more than one projection per file";
						return empty_return();
					}

					// read image data
					auto pitch = std::size_t{};
					auto img_buffer = std::unique_ptr<U, deleter_type>{Allocator::allocate(width, height, &pitch)};

					switch(header.type_of_numbers)
					{
						case tn_unsigned_char:
						{
							auto buffer = std::unique_ptr<std::uint8_t>{new std::uint8_t[width * height]};
							readEntry(file, buffer.get(), width * height * sizeof(std::uint8_t));
							readBuffer<U, std::uint8_t>(img_buffer.get(), buffer.get(), width, height);
							break;
						}

						case tn_unsigned_short:
						{
							auto buffer = std::unique_ptr<std::uint16_t>{new std::uint16_t[width * height]};
							readEntry(file, buffer.get(), width * height * sizeof(std::uint16_t));
							readBuffer<U, std::uint16_t>(img_buffer.get(), buffer.get(), width, height);
							break;
						}

						case tn_dword:
						{
							auto buffer = std::unique_ptr<std::uint32_t>{new std::uint32_t[width * height]};
							readEntry(file, buffer.get(), width * height * sizeof(std::uint32_t));
							readBuffer<U, std::uint32_t>(img_buffer.get(), buffer.get(), width, height);
							break;
						}

						case tn_double:
						{
							auto buffer = std::unique_ptr<double>{new double[width * height]};
							readEntry(file, buffer.get(), width * height * sizeof(double));
							readBuffer<U, double>(img_buffer.get(), buffer.get(), width, height);
							break;
						}

						case tn_float:
						{
							auto buffer = std::unique_ptr<float>{new float[width * height]};
							readEntry(file, buffer.get(), width * height * sizeof(float));
							readBuffer<U, float>(img_buffer.get(), buffer.get(), width, height);
							break;
						}

						default:
						{
							BOOST_LOG_TRIVIAL(warning) << "HIS loader: No implementation for data type of file " << path;
							return empty_return();
						}
					}

					return ddafa::image::Image<U, image_type>(width, height, std::move(img_buffer));
				}

			protected:
				~HIS() = default;

			private:
				template <typename U>
				inline auto readEntry(std::ifstream& file, U& entry) -> void
				{
					file.read(reinterpret_cast<char *>(&entry), sizeof(entry));
				}

				template <typename U>
				inline auto readEntry(std::ifstream& file, U* entry, std::size_t size) -> void
				{
					file.read(reinterpret_cast<char *>(entry), size);
				}

				template <typename Wanted, typename Actual>
				inline auto readBuffer(Wanted* dest, Actual* buf, std::uint32_t width, std::uint32_t height)
					-> typename std::enable_if<std::is_same<Wanted, Actual>::value>::type
				{
					for(std::size_t j = 0; j < height; ++j)
					{
						for(std::size_t i = 0; i < width; ++i)
							dest[i + j * width] = buf[i + j * width];
					}
				}

				template <typename Wanted, typename Actual>
				inline auto readBuffer(Wanted* dest, Actual* buf, std::uint32_t width, std::uint32_t height)
					-> typename std::enable_if<!std::is_same<Wanted, Actual>::value>::type
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
