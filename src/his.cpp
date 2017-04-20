/*
 * This file is part of the PARIS reconstruction program.
 *
 * Copyright (C) 2016 Helmholtz-Zentrum Dresden-Rossendorf
 *
 * PARIS is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PARIS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PARIS. If not, see <http://www.gnu.org/licenses/>.
 *
 * Date: 18 August 2016
 * Authors: Jan Stephan
 */

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#ifdef NO_BOOST_LOG
#include <iostream>
#else
#include <boost/log/trivial.hpp>
#endif

#include "backend.h"
#include "his.h"
#include "projection.h"

namespace paris
{
    namespace his
    {
        namespace
        {
            constexpr auto file_header_size     = 68;
            constexpr auto rest_size            = 34;
            constexpr auto hardware_header_size = 32;
            constexpr auto header_size          = file_header_size + hardware_header_size;
            constexpr auto file_id              = 0x7000;

            struct his_header
            {
                std::uint16_t file_type;            // = file_id
                std::uint16_t header_size;          // size of this file header in bytes
                std::uint16_t header_version;       // yy.y
                std::uint32_t file_size;            // size of the whole file in bytes
                std::uint16_t image_header_size;    // size of the image header in bytes
                std::uint16_t ulx, uly, brx, bry;   // bounding rectangle of image
                std::uint16_t frame_number;         // number of frames in the current file
                std::uint16_t correction;           // 0 = none, 1 = offset, 2 = gain, 4 = bad pixel, (ored)
                double integration_time;            // frame time in microseconds
                std::uint16_t number_type;          /* short, long integer, float, signed/unsigned, inverted,
                                                     * fault map, offset/gain correction data, badpixel correction data
                                                     * */
                std::uint8_t x[rest_size];          // fill up to 68 bytes
            };

            enum class data
            {
                type_not_implemented    = -1,
                type_uchar              = 2,
                type_ushort             = 4,
                type_dword              = 32,
                type_double             = 64,
                type_float              = 128
            };

            template <typename U>
            auto read_entry(std::ifstream& file, U& entry) -> void
            {
                using char_type = typename std::ifstream::char_type;
                file.read(reinterpret_cast<char_type*>(&entry), sizeof(entry));
            }

            template <typename U>
            auto read_entry(std::ifstream& file, U* entry, std::streamsize size) -> void
            {
                using char_type = typename std::ifstream::char_type;
                file.read(reinterpret_cast<char_type*>(entry), size);
            }

            template <typename T>
            auto copy_to_buf(std::ifstream& file, float* dest, std::uint16_t w, std::uint16_t h) -> void
            {
                auto w_s = static_cast<std::size_t>(w);
                auto h_s = static_cast<std::size_t>(h);
                auto size = static_cast<std::streamsize>(w_s * h_s * sizeof(T));
                auto buffer = std::unique_ptr<T[]>{new T[w_s * h_s]};
                read_entry(file, buffer.get(), size);
                std::copy(buffer.get(), buffer.get() + (w_s * h_s), dest);
            }
        }

        auto load(const std::string& path) -> std::vector<image_type>
        {
            auto vec = std::vector<image_type>{};

            auto header = his_header{};

            auto&& file = std::ifstream{path.c_str(), std::ios_base::binary};
            if(!file.is_open())
            {
#ifdef NO_BOOST_LOG
                std::cout << "his_loader::load() failed to open file at " << path << "\n";
#else
                BOOST_LOG_TRIVIAL(warning) << "his_loader::load() failed to open file at " << path;
#endif
                throw std::system_error{errno, std::generic_category()};
            }

            read_entry(file, header.file_type);
            read_entry(file, header.header_size);
            read_entry(file, header.header_version);
            read_entry(file, header.file_size);
            read_entry(file, header.image_header_size);
            read_entry(file, header.ulx);
            read_entry(file, header.uly);
            read_entry(file, header.brx);
            read_entry(file, header.bry);
            read_entry(file, header.frame_number);
            read_entry(file, header.correction);
            read_entry(file, header.integration_time);
            read_entry(file, header.number_type);
            read_entry(file, header.x);

            if(header.file_type != file_id)
            {
#ifdef NO_BOOST_LOG
                std::cout << "his_loader::load() could not open non-HIS file at " << path << "\n";
#else
                BOOST_LOG_TRIVIAL(warning) << "his_loader::load() could not open non-HIS file at " << path;
#endif
                return vec;
            }
            if(header.header_size != file_header_size)
            {
#ifdef NO_BOOST_LOG
                std::cout << "his_loader::load() encountered a file header size mismatch at " << path << "\n";
#else
                BOOST_LOG_TRIVIAL(warning) << "his_loader::load() encountered a file header size mismatch at " << path;
#endif
                return vec;
            }
            if(header.number_type == static_cast<std::uint16_t>(data::type_not_implemented))
            {
#ifdef NO_BOOST_LOG
                std::cout << "his_loader::load() encountered an unsupported data type at " << path << "\n";
#else
                BOOST_LOG_TRIVIAL(warning) << "his_loader::load() encountered an unsupported data type at " << path;
#endif
                return vec;
            }

            auto x1 = static_cast<std::uint32_t>(header.ulx);
            auto x2 = static_cast<std::uint32_t>(header.brx);
            auto y1 = static_cast<std::uint32_t>(header.uly);
            auto y2 = static_cast<std::uint32_t>(header.bry);
            auto width = x2 - x1 + 1u;
            auto height = y2 - y1 + 1u;
            for(auto i = 0u; i < header.frame_number; ++i)
            {
                // skip image header
                auto img_header = std::unique_ptr<std::uint8_t[]>{new std::uint8_t[header.image_header_size]};
                read_entry(file, img_header.get(), header.image_header_size);

                auto img = backend::make_projection_host(width, height);

                auto w16 = static_cast<std::uint16_t>(width);
                auto h16 = static_cast<std::uint16_t>(height);
                using num_type = decltype(header.number_type);
                switch(header.number_type)
                {
                    case static_cast<num_type>(data::type_uchar):
                        copy_to_buf<std::uint8_t>(file, img.buf.get(), w16, h16);
                        break;

                    case static_cast<num_type>(data::type_ushort):
                        copy_to_buf<std::uint16_t>(file, img.buf.get(), w16, h16);
                        break;

                    case static_cast<num_type>(data::type_dword):
                        copy_to_buf<std::uint32_t>(file, img.buf.get(), w16, h16);
                        break;

                    case static_cast<num_type>(data::type_double):
                        copy_to_buf<double>(file, img.buf.get(), w16, h16);
                        break;

                    case static_cast<num_type>(data::type_float):
                        copy_to_buf<float>(file, img.buf.get(), w16, h16);
                        break;

                    default:
#ifdef NO_BOOST_LOG
                        std::cout << "his_loader::load() tried to load an unsupported data type.\n";
#else
                        BOOST_LOG_TRIVIAL(warning) << "his_loader::load() tried to load an unsupported data type.";
#endif
                        return vec;
                }

                img.dim_x = static_cast<std::uint32_t>(width);
                img.dim_y = static_cast<std::uint32_t>(height);
                vec.push_back(std::move(img));
            }
            return vec;
        }
    }
}

