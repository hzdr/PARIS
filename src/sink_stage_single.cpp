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
#include <stdexcept>
#include <utility>

#include <boost/log/trivial.hpp>

#include <cuda_runtime.h>

#include <ddrf/cuda/memory.h>

#include "exception.h"
#include "filesystem.h"
#include "metadata.h"
#include "sink_stage_single.h"
#include "tiff_saver_single.h"

namespace ddafa
{
    sink_stage_single::sink_stage_single(const std::string& path, const std::string& prefix)
    : path_{path}
    {
        if(path_.back() != '/')
            path_ += '/';
        path_ += prefix;

        try
        {
            auto s = create_directory(path);
            if(!s)
            {
                BOOST_LOG_TRIVIAL(fatal) << "sink_stage_single::sink_stage_single() failed to create output directory at " << path;
                throw stage_construction_error{"sink_stage_single::sink_stage_single failed()"};
            }
        }
        catch(const std::runtime_error& re)
        {
            BOOST_LOG_TRIVIAL(fatal) << "sink_stage_single::sink_stage_single() encountered a runtime error during directory creation at " << path;
            throw stage_construction_error{"sink_stage_single::sink_stage_single failed()"};
        }
    }

    sink_stage_single::sink_stage_single(sink_stage_single&& other) noexcept
    : input_{std::move(other.input_)}, devices_{std::move(other.devices_)}
    , path_{std::move(other.path_)}, prefix_{std::move(other.prefix_)}
    {
        other.moved_ = true;
    }

    auto sink_stage_single::operator=(sink_stage_single&& other) noexcept -> sink_stage_single&
    {
        input_ = std::move(other.input_);
        devices_ = std::move(other.devices_);
        path_ = std::move(other.path_);
        prefix_ = std::move(other.prefix_);

        other.moved_ = true;

        return *this;
    }

    sink_stage_single::~sink_stage_single()
    {
        if(!moved_)
        {
            for(auto i = 0; i < devices_; ++i)
            {
                cudaSetDevice(i);
                cudaDeviceReset();
            }
        }
    }

    auto sink_stage_single::run() -> void
    {
        try
        {
            auto i = std::size_t{0};
            auto saver = tiff_saver_single{};
            while(true)
            {
                auto proj = input_();
                if(!proj.second.valid)
                    break;

                auto proj_path = path_ + std::to_string(i);
                saver.save(std::move(proj), proj_path);
                ++i;
            }
        }
        catch(const std::runtime_error re)
        {
            BOOST_LOG_TRIVIAL(fatal) << "sink_stage_single::run() failed to save projection: " << re.what();
            throw stage_runtime_error{"sink_stage_single::run() failed"};
        }
    }

    auto sink_stage_single::set_input_function(std::function<input_type(void)> input) noexcept -> void
    {
        input_ = input;
    }
}
