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
#include <utility>

#include <boost/log/trivial.hpp>

#include <ddrf/cuda/algorithm.h>
#include <ddrf/cuda/memory.h>
#include <ddrf/cuda/sync_policy.h>
#include <ddrf/memory.h>

#include "device_to_host_stage.h"
#include "exception.h"
#include "metadata.h"

namespace ddafa
{
    auto device_to_host_stage::run() -> void
    {
        try
        {
            while(true)
            {
                auto d_proj = input_();
                if(!d_proj.second.valid)
                    break;

                auto h_proj = ddrf::cuda::make_unique_pinned_host<float>(d_proj.second.width, d_proj.second.height);
                ddrf::cuda::copy(ddrf::cuda::sync, h_proj, d_proj.first, d_proj.second.width, d_proj.second.height);
                output_(std::make_pair(std::move(h_proj), d_proj.second));
            }
            output_(output_type());
        }
        catch(const ddrf::cuda::bad_alloc& ba)
        {
            BOOST_LOG_TRIVIAL(fatal) << "device_to_host_stage::run() failed to allocate memory: " << ba.what();
            throw stage_runtime_error{"device_to_host_stage::run() failed"};
        }
    }

    auto device_to_host_stage::set_input_function(std::function<input_type(void)> input) noexcept -> void
    {
        input_ = input;
    }

    auto device_to_host_stage::set_output_function(std::function<void(output_type)> output) noexcept -> void
    {
        output_ = output;
    }
}
