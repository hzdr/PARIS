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
 * Date: 21 November 2016
 * Authors: Jan Stephan <j.stephan@hzdr.de>
 */

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <fstream>
#include <ios>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

#include <boost/log/trivial.hpp>

#include "ddbvf.h"
#include "volume.h"

namespace paris
{
    namespace ddbvf
    {
        namespace
        {
            constexpr auto ddbvf_id = 0xEFDDDAFA;
            constexpr auto ddbvf_version = 0x0010;

            // as all types are the same we don't need to consider padding here
            struct header
            {
                std::uint32_t dim_x;
                std::uint32_t dim_y;
                std::uint32_t dim_z;
                std::uint32_t offset;
            };
            
            constexpr auto offset_pos = sizeof(ddbvf_id) + sizeof(ddbvf_version) + sizeof(header) - sizeof(header::offset);
            constexpr auto first_pos = 32;
        }

        struct handle
        {
            header head;
            std::fstream stream;
        };

        auto handle_deleter::operator()(handle* h) noexcept -> void
        {
            delete h;
        }

        auto create(const std::string& path, std::uint32_t dim_x, std::uint32_t dim_y, std::uint32_t dim_z) -> handle_type
        {
            auto full_path = path + ".ddbvf";

            auto h = handle_type{new handle};
            h->head = {dim_x, dim_y, dim_z, 0u};

            // the first 32 bytes are reserved for the file header
            h->head.offset = first_pos - sizeof(ddbvf_id) - sizeof(ddbvf_version) - sizeof(h->head);

            h->stream.open(full_path.c_str(), std::ios::out | std::ios::binary | std::ios::trunc);

            // write file header
            h->stream.write(reinterpret_cast<const char*>(&ddbvf_id), sizeof(ddbvf_id));
            h->stream.write(reinterpret_cast<const char*>(&ddbvf_version), sizeof(ddbvf_version));
            h->stream.write(reinterpret_cast<char*>(&h->head), sizeof(h->head));
            
            // fill remaining bytes with zeroes
            auto buf = std::unique_ptr<char[]>{new char[h->head.offset]};
            std::fill_n(buf.get(), h->head.offset, 0);
            h->stream.write(buf.get(), static_cast<std::streamsize>(h->head.offset));

            // check for errors
            if(!h->stream)
                throw std::system_error{errno, std::generic_category()};

            return h;
        }

        auto open(const std::string& path) -> handle_type
        {
            auto h = handle_type{new handle};

            h->stream.open(path.c_str(), std::ios::in | std::ios::out | std::ios::binary);

            auto id = std::uint32_t{};
            auto version = std::uint16_t{};
            
            // read file header
            h->stream.read(reinterpret_cast<char*>(&id), sizeof(id));
            if(id != ddbvf_id)
                throw std::runtime_error{"Not a ddbvf file: " + path};

            h->stream.read(reinterpret_cast<char*>(&version), sizeof(version));
            if(version != ddbvf_version)
                throw std::runtime_error{"Unsupported ddbvf version: " + path};

            h->stream.read(reinterpret_cast<char*>(&h->head), sizeof(h->head));

            // check for errors
            if(!h->stream)
                throw std::system_error{errno, std::generic_category()};

            return h;
        }

        auto write(handle_type& h, const volume_type& vol, std::uint32_t first) -> void
        {
            if(h == nullptr || vol.buf == nullptr)
                return;

            // check dimensions
            if(first >= h->head.dim_z)
                throw std::runtime_error{"ddbvf::write(): Starting position out of bounds"};

            if(vol.dim_x != h->head.dim_x || vol.dim_y != h->head.dim_y || vol.dim_z > h->head.dim_z)
                throw std::runtime_error{"ddbvf::write(): Attempting to save volume to file with wrong dimensions"};

            // calculate size and offset for writing
            using element_type = typename decltype(volume_type::buf)::element_type;
            auto write_size = static_cast<std::streamsize>(vol.dim_x * vol.dim_y * vol.dim_z * sizeof(element_type));
            auto write_pos = static_cast<std::fstream::off_type>(vol.dim_x * vol.dim_y * first * sizeof(element_type));

            // write data
            h->stream.seekp(first_pos);
            h->stream.seekp(write_pos, std::ios_base::cur);
            h->stream.write(reinterpret_cast<char*>(vol.buf.get()), write_size);

            // check for errors
            if(!h->stream)
                throw std::system_error{errno, std::generic_category()};
        }
    }
}
