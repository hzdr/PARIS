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
#include <cstdlib>
#include <functional>
#include <utility>
#include <vector>

#include <boost/log/trivial.hpp>

#include <ddrf/cuda/algorithm.h>
#include <ddrf/cuda/coordinates.h>
#include <ddrf/cuda/launch.h>
#include <ddrf/cuda/sync_policy.h>
#include <ddrf/cuda/utility.h>

#include "exception.h"
#include "projection.h"
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

                dst_row[x] = 0.f;
            }
        }
    }

    preloader_stage::preloader_stage(std::size_t pool_limit)
    : pools_{}
    {
        auto sce = stage_construction_error{"preloader_stage::preloader_stage() failed"};

        try
        {
            devices_ = ddrf::cuda::get_device_count();
            for(auto i = 0; i < devices_; ++i)
                pools_.emplace_back(pool_limit);
        }
        catch(const ddrf::cuda::bad_alloc& ba)
        {
            BOOST_LOG_TRIVIAL(fatal) << "preloader_stage::preloader_stage() encountered a bad_alloc: " << ba.what();
            throw sce;
        }
        catch(const ddrf::cuda::invalid_argument& ia)
        {
            BOOST_LOG_TRIVIAL(fatal) << "preloader_stage::preloader_stage() encountered an invalid argument error: " << ia.what();
            throw sce;
        }
        catch(const ddrf::cuda::runtime_error& re)
        {
            BOOST_LOG_TRIVIAL(fatal) << "preloader_stage::preloader_stage() caused a runtime error: " << re.what();
            throw sce;
        }
    }

    preloader_stage::~preloader_stage()
    {
        if(pools_.empty())
            return;

        for(auto i = 0; i < devices_; ++i)
        {
            auto err = cudaSetDevice(i); // ddrf::cuda::set_device() can throw, cudaSetDevice() does not
            if(err != cudaSuccess)
            {
                BOOST_LOG_TRIVIAL(fatal) << "preloader_stage::~preloader_stage() encountered error: " << cudaGetErrorString(err);
                std::exit(err);
            }

            auto d = static_cast<p_size_type>(i);
            pools_.at(d).release();
        }
}

    auto preloader_stage::run() -> void
    {
        auto sre = stage_runtime_error{"preloader_stage::run() failed"};
        try
        {
            while(true)
            {
                auto proj = input_();

                if(!proj.valid)
                    break;

                for(auto i = 0; i < devices_; ++i)
                {
                    ddrf::cuda::set_device(i);

                    auto d_v = static_cast<p_size_type>(i);
                    auto&& alloc = pools_[d_v];
                    auto dev_proj = alloc.allocate_smart(proj.width, proj.height);

                    auto stream = ddrf::cuda::create_stream();

                    // we have to initialize the destination data before copying because of reasons
                    ddrf::cuda::launch(proj.width, proj.height, init_kernel,
                            dev_proj.get(), proj.width, proj.height, dev_proj.pitch());

                    ddrf::cuda::copy(ddrf::cuda::sync, dev_proj, proj.ptr, proj.width, proj.height);

                    output_(output_type{std::move(dev_proj), proj.width, proj.height, proj.idx, proj.phi, true, i, ddrf::cuda::create_stream()});
                }
            }

            // Uploaded all projections to the GPU, notify the next stage that we are done here
            output_(output_type{});
            BOOST_LOG_TRIVIAL(info) << "Uploaded all projections to the device(s)";
        }
        catch(const ddrf::cuda::bad_alloc& ba)
        {
            BOOST_LOG_TRIVIAL(fatal) << "preloader_stage::run() encountered a bad_alloc: " << ba.what();
            throw sre;
        }
        catch(const ddrf::cuda::invalid_argument& ia)
        {
            BOOST_LOG_TRIVIAL(fatal) << "preloader_stage::run() passed an invalid argument to the CUDA runtime: " << ia.what();
            throw sre;
        }
        catch(const ddrf::cuda::runtime_error& re)
        {
            BOOST_LOG_TRIVIAL(fatal) << "preloader_stage::run() caused a CUDA runtime error: " << re.what();
            throw sre;
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
