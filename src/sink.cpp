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
 * Date: 19 August 2016
 * Authors: Jan Stephan <j.stephan@hzdr.de>
 */

#include <cstddef>
#include <stdexcept>
#include <mutex>
#include <utility>

#include <boost/log/trivial.hpp>

#include "backend.h"
#include "exception.h"
#include "filesystem.h"
#include "sink.h"
#include "ddbvf.h"
#include "volume.h"

namespace paris
{
    sink::sink(const std::string& path, const std::string& prefix, const volume_geometry& vol_geo)
    : path_{path}, vol_geo_(vol_geo)
    {
        try
        {
            if(path_.back() != '/')
                path_ += '/';
            path_ += prefix;

            auto s = create_directory(path);
            if(!s)
            {
                BOOST_LOG_TRIVIAL(fatal) << "sink::sink() failed to create output directory at " << path;
                throw stage_construction_error{"sink::sink() failed"};
            }

            handle_ = ddbvf::create(path_, vol_geo_.dim_x, vol_geo_.dim_y, vol_geo_.dim_z);
        }
        catch(const std::system_error& se)
        {
            BOOST_LOG_TRIVIAL(fatal) << "sink::sink(): system error while creating volume: "
                                        << se.code() << " - " << se.what();
            throw stage_runtime_error{"sink::sink() failed"};
        }
        catch(const std::runtime_error& re)
        {
            BOOST_LOG_TRIVIAL(fatal) << "sink::sink() encountered a runtime error";
            BOOST_LOG_TRIVIAL(fatal) << re.what();
            throw stage_construction_error{"sink::sink() failed"};
        }

    }

    auto sink::save(const backend::volume_device_type& v) -> void
    {
        try
        {
            auto host_v = backend::make_volume_host(v.dim_x, v.dim_y, v.dim_z);
            backend::copy_d2h(v, host_v);

            static auto&& m = std::mutex{};
            auto&& lock = std::lock_guard<std::mutex>{m};
            ddbvf::write(handle_, host_v, host_v.off);
        }
        catch(const std::system_error& se)
        {
            BOOST_LOG_TRIVIAL(fatal) << "sink::save(): system error while saving volume: "
                                        << se.code() << " - " << se.what();
            throw stage_runtime_error{"sink::save() failed"};
        }
        catch(const std::runtime_error& re)
        {
            BOOST_LOG_TRIVIAL(fatal) << "sink::save(): runtime error while saving volume: " << re.what();
            throw stage_runtime_error{"sink::save() failed"};
        }
    }
}
