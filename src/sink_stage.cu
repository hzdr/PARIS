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
#include "task.h"
#include "ddbvf.h"
#include "volume.h"

namespace ddafa
{
    sink_stage::sink_stage(const std::string& path, const std::string& prefix, const volume_geometry& vol_geo, int devices)
    : path_{path}, vol_geo_(vol_geo), devices_{devices}
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

    auto sink_stage::assign_task(task t) noexcept -> void
    {
    }

    auto sink_stage::run() -> void
    {
        try
        {
            auto handle = ddbvf::create(path_, vol_geo_.dim_x, vol_geo_.dim_y, vol_geo_.dim_z);
            auto count = devices_;
            while(true)
            {
                auto vol = input_();
                --count;
                ddbvf::write(handle, vol, vol.offset);
                if(count == 0)
                    break;
            }
        }
        catch(const std::system_error& se)
        {
            BOOST_LOG_TRIVIAL(fatal) << "sink_stage::run(): system error while saving volume: "
                                        << se.code() << " - " << se.what();
            throw stage_runtime_error{"sink_stage::run() failed"};
        }
        catch(const std::runtime_error& re)
        {
            BOOST_LOG_TRIVIAL(fatal) << "sink_stage::run(): runtime error while saving volume: " << re.what();
            throw stage_runtime_error{"sink_stage::run() failed"};
        }
    }

    auto sink_stage::set_input_function(std::function<input_type(void)> input) noexcept -> void
    {
        input_ = input;
    }
}
