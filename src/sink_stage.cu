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
 * Date: 19 August 2016
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
#include "sink_stage.h"
#include "tiff_saver.h"
#include "volume.h"

namespace ddafa
{
    sink_stage::sink_stage(const std::string& path, const std::string& prefix)
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
                BOOST_LOG_TRIVIAL(fatal) << "sink_stage::sink_stage() failed to create output directory at " << path;
                throw stage_construction_error{"sink_stage::sink_stage failed()"};
            }
        }
        catch(const std::runtime_error& re)
        {
            BOOST_LOG_TRIVIAL(fatal) << "sink_stage::sink_stage() encountered a runtime error during directory creation at " << path;
            throw stage_construction_error{"sink_stage::sink_stage failed()"};
        }
    }

    auto sink_stage::run() -> void
    {
        try
        {
            auto vol = input_();
            auto saver = tiff_saver{};
            saver.save(std::move(vol), path_);
        }
        catch(const std::runtime_error re)
        {
            BOOST_LOG_TRIVIAL(fatal) << "sink_stage::run() failed to save volume: " << re.what();
            throw stage_runtime_error{"sink_stage::run() failed"};
        }
    }

    auto sink_stage::set_input_function(std::function<input_type(void)> input) noexcept -> void
    {
        input_ = input;
    }
}
