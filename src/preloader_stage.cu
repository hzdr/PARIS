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

#include <cstddef>
#include <functional>
#include <utility>
#include <vector>

#include <boost/log/trivial.hpp>

#include <ddrf/cuda/algorithm.h>
#include <ddrf/cuda/coordinates.h>
#include <ddrf/cuda/launch.h>
#include <ddrf/cuda/sync_policy.h>

#include "exception.h"
#include "metadata.h"
#include "preloader_stage.h"

namespace ddafa
{
    namespace
    {
        __global__ void init_kernel(float* dst, std::size_t w, std::size_t h, std::size_t p)
        {
            auto x = ddrf::cuda::coord_x();
            auto y = ddrf::cuda::coord_y();

            if((x < w) && (y < h))
            {
                auto dst_row = reinterpret_cast<float*>(reinterpret_cast<char*>(dst) + p * y);

                dst_row[x] = 1.f;
            }
        }
    }

    preloader_stage::preloader_stage(std::size_t pool_limit)
    : pools_{}, moved_{false}
    {
        auto err = cudaGetDeviceCount(&devices_);
        if(err != cudaSuccess)
        {
            BOOST_LOG_TRIVIAL(fatal) << "preloader_stage::preloader_stage() could not obtain CUDA devices: " << cudaGetErrorString(err);
            throw stage_construction_error{"preloader_stage::preloader_stage() failed"};
        }

        for(auto i = 0; i < devices_; ++i)
            pools_.emplace_back(pool_limit);
    }

    preloader_stage::~preloader_stage()
    {
        if(pools_.empty())
            return;

        for(auto i = 0; i < devices_; ++i)
        {
            cudaSetDevice(i);
            using size_type = typename decltype(pools_)::size_type;
            auto d = static_cast<size_type>(i);
            pools_[d].release();
        }
}

    auto preloader_stage::run() -> void
    {
        try
        {
            BOOST_LOG_TRIVIAL(debug) << "Called preloader_stage::run()";
            while(true)
            {
                auto proj = input_();

                if(!proj.second.valid)
                    break;

                for(auto i = 0; i < devices_; ++i)
                {
                    auto err = cudaSetDevice(i);
                    if(err != cudaSuccess)
                    {
                        BOOST_LOG_TRIVIAL(fatal) << "preloader_stage::run() could not set CUDA device: " << cudaGetErrorString(err);
                        throw stage_runtime_error{"preloader_stage::run() failed to initialize"};
                    }

                    using size_type = typename decltype(pools_)::size_type;
                    auto d_v = static_cast<size_type>(i);
                    auto&& alloc = pools_[d_v];
                    auto dev_proj = alloc.allocate_smart(proj.second.width, proj.second.height);

                    // we have to initialize the destination data before copying because of reasons
                    ddrf::cuda::launch(proj.second.width, proj.second.height, init_kernel,
                            dev_proj.get(), proj.second.width, proj.second.height, dev_proj.pitch());

                    ddrf::cuda::copy(ddrf::cuda::sync, dev_proj, proj.first, proj.second.width, proj.second.height);

                    auto meta = projection_metadata{proj.second.width, proj.second.height, proj.second.index, proj.second.phi, true, i};
                    output_(std::make_pair(std::move(dev_proj), meta));
                }
            }

            // Uploaded all projections to the GPU, notify the next stage that we are done here
            output_(std::make_pair(nullptr, projection_metadata{0, 0, 0, 0.f, false, 0}));
            BOOST_LOG_TRIVIAL(info) << "Uploaded all projections to the device(s)";
        }
        catch(const ddrf::cuda::invalid_argument& ia)
        {
            BOOST_LOG_TRIVIAL(fatal) << "preloader_stage::run() could not copy to CUDA device: " << ia.what();
            throw stage_runtime_error{"preloader_stage::run() failed to copy projection"};
        }
        catch(const ddrf::cuda::bad_alloc& ba)
        {
            BOOST_LOG_TRIVIAL(fatal) << "preloader_stage::run() encountered a bad_alloc: " << ba.what();
            throw stage_runtime_error{"preloader_stage::run() failed"};
        }
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
