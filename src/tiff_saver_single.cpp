/*
 * This file is part of the ddafa reconstruction program.
 *
 * Copyright (C) 2016 Helmholtz-Zentrum Dresden-Rossendorf
 *
 * Licensed under the EUPL, Version 1.1 or - as soon they will be approved by
 * the European Commission - subsequent version of the EUPL (the "Licence");
 * You may not use this work except in compliance with the Licence.
 * You may obtain a copy of the Licence at:
 *
 * http://ec.europa.eu/idabc/eupl
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the Licence is distributed on an "AS IS" basis,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the Licence for the specific language governing permissions and
 * limitations under the Licence.
 *
 * Date: 24 August 2016
 * Authors: Jan Stephan
 */

#include <functional>
#include <locale>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include <boost/date_time/posix_time/posix_time.hpp>

#include <tiffio.h>

#include <ddrf/cuda/memory.h>

#include "metadata.h"
#include "tiff_saver_single.h"

namespace ddafa
{
    template <class T, bool = std::is_integral<T>::value, bool = std::is_unsigned<T>::value> struct sample_format {};
    template <class T> struct sample_format<T, true, true> { static constexpr auto value = SAMPLEFORMAT_UINT; };
    template <class T> struct sample_format<T, true, false> { static constexpr auto value = SAMPLEFORMAT_INT; };
    template <> struct sample_format<float> { static constexpr auto value = SAMPLEFORMAT_IEEEFP; };
    template <> struct sample_format<double> { static constexpr auto value = SAMPLEFORMAT_IEEEFP; };

    template <class T> struct bits_per_sample { static constexpr auto value = (sizeof(T) * 8); };

    auto tiff_saver_single::save(std::pair<ddrf::cuda::pinned_host_ptr<float>, projection_metadata> proj, const std::string& path) const -> void
    {
        auto full_path = path;
        full_path.append(".tif");

        auto tif = std::unique_ptr<TIFF, std::function<void(TIFF*)>>{TIFFOpen(full_path.c_str(), "w8"), [](TIFF* p) { TIFFClose(p); }};
        if(tif == nullptr)
            throw std::runtime_error{"tiff_saver::save() failed to open " + full_path + " for writing"};

        auto&& ss = std::stringstream{};
        // the locale will take ownership so plain new is okay here
        auto output_facet = new boost::posix_time::time_facet{"%Y:%m:%d %H:%M:%S"};

        ss.imbue(std::locale{std::locale::classic(), output_facet});
        ss.str("");

        auto now = boost::posix_time::second_clock::local_time();
        ss << now;

        auto tifp = tif.get();
        TIFFSetField(tifp, TIFFTAG_IMAGEWIDTH, proj.second.width);
        TIFFSetField(tifp, TIFFTAG_IMAGELENGTH, proj.second.height);
        TIFFSetField(tifp, TIFFTAG_BITSPERSAMPLE, bits_per_sample<float>::value);
        TIFFSetField(tifp, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
        TIFFSetField(tifp, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
        TIFFSetField(tifp, TIFFTAG_THRESHHOLDING, THRESHHOLD_BILEVEL);
        TIFFSetField(tifp, TIFFTAG_SAMPLESPERPIXEL, 1);
        TIFFSetField(tifp, TIFFTAG_SOFTWARE, "ddafa");
        TIFFSetField(tifp, TIFFTAG_DATETIME, ss.str().c_str());

        TIFFSetField(tifp, TIFFTAG_SAMPLEFORMAT, sample_format<float>::value);

        auto data_ptr = proj.first.get();
        for(auto row = 0u; row < proj.second.height; ++row)
        {
            TIFFWriteScanline(tifp, reinterpret_cast<void*>(data_ptr), row);
            data_ptr += proj.second.width;
        }
    }
}
