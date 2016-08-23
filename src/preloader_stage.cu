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
 * Date: 18 August 2016
 * Authors: Jan Stephan
 */

#include <functional>
#include <utility>
#include <vector>

#include <boost/log/trivial.hpp>

#include <ddrf/cuda/algorithm.h>
#include <ddrf/cuda/sync_policy.h>

#include "exception.h"
#include "metadata.h"
#include "preloader_stage.h"

namespace ddafa
{
    auto preloader_stage::run() -> void
    {
        BOOST_LOG_TRIVIAL(debug) << "Called preloader_stage::run()";
        auto devices = int{};
        auto err = cudaGetDeviceCount(&devices);
        if(err != cudaSuccess)
        {
            BOOST_LOG_TRIVIAL(fatal) << "preloader_stage::run() could not obtain CUDA devices: " << cudaGetErrorString(err);
            throw stage_runtime_error{"preloader_stage::run() failed to initialize"};
        }

        using vec_type = std::vector<pool_allocator>;
        using v_size_type = typename vec_type::size_type;
        auto d_v = static_cast<v_size_type>(devices);
        auto pools = std::vector<pool_allocator>{d_v};

        while(true)
        {
            auto proj = input_();

            if(!proj.second.valid)
                break;

            for(auto i = 0; i < devices; ++i)
            {
                err = cudaSetDevice(i);
                if(err != cudaSuccess)
                {
                    BOOST_LOG_TRIVIAL(fatal) << "preloader_stage::run() could not set CUDA device: " << cudaGetErrorString(err);
                    throw stage_runtime_error{"preloader_stage::run() failed to initialize"};
                }

                d_v = static_cast<v_size_type>(i);
                auto& alloc = pools[d_v];
                auto dev_proj = alloc.allocate_smart(proj.second.width, proj.second.height);
                try
                {
                    ddrf::cuda::copy(ddrf::cuda::async, dev_proj, proj.first, proj.second.width, proj.second.height);
                }
                catch(const ddrf::cuda::invalid_argument& ia)
                {
                    BOOST_LOG_TRIVIAL(fatal) << "preloader_stage::run() could not copy to CUDA device: " << ia.what();
                    throw stage_runtime_error{"preloader_stage::run() failed to copy projection"};
                }

                auto meta = projection_metadata{proj.second.width, proj.second.height, proj.second.index, proj.second.phi, true, i};
                output_(std::make_pair(std::move(dev_proj), meta));
            }
        }

        // Uploaded all projections to the GPU, notify the next stage that we are done here
        output_(std::make_pair(nullptr, projection_metadata{0, 0, 0, 0.f, false, 0}));
        BOOST_LOG_TRIVIAL(info) << "Uploaded all projections to the device(s)";
    }

    auto preloader_stage::set_input_function(std::function<input_type(void)> input) noexcept -> void
    {
        input_ = input;
    }

    auto preloader_stage::set_output_function(std::function<void(output_type)> output) noexcept -> void
    {
        output_ = output;
    }
}
