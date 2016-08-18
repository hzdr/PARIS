#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <fstream>
#include <ios>
#include <memory>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <boost/log/trivial.hpp>

#include "his_loader.h"

namespace ddafa
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
        auto read_entry(std::ifstream& file, U& entry)
        {
            using char_type = typename std::ifstream::char_type;
            file.read(reinterpret_cast<char_type*>(&entry), sizeof(entry));
        }

        template <typename U>
        auto read_entry(std::ifstream& file, U* entry, std::streamsize size)
        {
            using char_type = typename std::ifstream::char_type;
            file.read(reinterpret_cast<char_type*>(entry), size);
        }

        template <typename T>
        auto copy_to_buf(std::ifstream& file, float* dest, std::uint16_t w, std::uint16_t h)
        {
            auto buffer = std::unique_ptr<T[]>{new T[w * h]};
            readEntry(file, buffer.get(), w * h * sizeof(T));
            std::copy(buffer.get(), buffer.get() + (w * h), dest);
        }
    }

    auto his_loader::load(const std::string& path) -> std::vector<image_type>
    {
        auto vec = std::vector<image_type>{};

        auto header = his_header{};

        auto&& file = std::ifstream{path.c_str(), std::ios_base::binary};
        if(!file.is_open())
        {
            BOOST_LOG_TRIVIAL(warning) << "his_loader::load() failed to open file at " << path;
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
            BOOST_LOG_TRIVIAL(warning) << "his_loader::load() could not open non-HIS file at " << path;
            throw std::runtime_error{"Invalid file type"};
        }
        if(header.header_size != file_header_size)
        {
            BOOST_LOG_TRIVIAL(warning) << "his_loader::load() encountered a file header size mismatch at " << path;
            throw std::runtime_error{"File header size mismatch"};
        }
        if(header.number_type == data::type_not_implemented)
        {
            BOOST_LOG_TRIVIAL(warning) << "his_loader::load() encountered an unsupported data type at " << path;
            throw std::runtime_error{"File with unsupported data type"};
        }

        auto width = header.brx - header.ulx + 1;
        auto height = header.bry - header.uly + 1;
        for(auto i = 0u; i < header.frame_number; ++i)
        {
            // skip image header
            auto img_header = std::unique_ptr<std::uint8_t[]>{new std::uint8_t[header.image_header_size]};
            readEntry(file, img_header.get(), header.image_header_size);

            auto img_buffer = alloc_.smart_allocate(width, height);

            using num_type = decltype(header.number_type);
            switch(header.number_type)
            {
                case static_cast<num_type>(data::type_uchar):
                    copy_to_buf<std::uint8_t>(file, img_buffer.get(), width, height);
                    break;

                case static_cast<num_type>(data::type_ushort):
                    copy_to_buf<std::uint16_t>(file, img_buffer.get(), width, height);
                    break;

                case static_cast<num_type>(data::type_dword):
                    copy_to_buf<std::uint32_t>(file, img_buffer.get(), width, height);
                    break;

                case static_cast<num_type>(data::type_double):
                    copy_to_buf<double>(file, img_buffer.get(), width, height);
                    break;

                case static_cast<num_type>(data::type_float):
                    copy_to_buf<float>(file, img_buffer.get(), width, height);
                    break;

                default:
                    BOOST_LOG_TRIVIAL(warning) << "his_loader::load() tried to load an unsupported data type."
                    throw std::runtime_error{"File with unsupported data type"};
            }

            auto m = projection_metadata{width, height, 0, 0, true};
            vec.emplace_back(std::make_pair(std::move(img_buffer, projection_metadata{width, height, 0, 0, true, 0})));
        }
        return vec;
    }
}

