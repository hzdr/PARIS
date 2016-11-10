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

#include <boost/log/trivial.hpp>

#include <ddrf/cuda/algorithm.h>
#include <ddrf/cuda/coordinates.h>
#include <ddrf/cuda/launch.h>
#include <ddrf/cuda/sync_policy.h>
#include <ddrf/cuda/utility.h>

#include "exception.h"
#include "projection.h"
#include "preloader_stage.h"
#include "task.h"

namespace ddafa
{
    preloader_stage::preloader_stage(std::size_t pool_limit, int device) noexcept
    : device_{device}, pool_{pool_limit}
    {}

    preloader_stage::~preloader_stage()
    {
        auto err = cudaSetDevice(device_); // ddrf::cuda::set_device() can throw, cudaSetDevice() does not
        if(err != cudaSuccess)
        {
            BOOST_LOG_TRIVIAL(fatal) << "preloader_stage::~preloader_stage() encountered error: " << cudaGetErrorString(err);
            std::exit(err);
        }

        pool_.release();
    }

    auto preloader_stage::assign_task(task t) noexcept -> void
    {}

    auto preloader_stage::run() -> void
    {
        auto sre = stage_runtime_error{"preloader_stage::run() failed"};
        try
        {
            ddrf::cuda::set_device(device_);
            while(true)
            {

                auto p = input_();

                if(!p.valid)
                    break;

                auto dev_p = pool_.allocate_smart(p.width, p.height);
                auto stream = ddrf::cuda::create_concurrent_stream();

                // initialize the destination data before copying
                ddrf::cuda::fill(ddrf::cuda::async, dev_p, 0, stream, p.width, p.height);
                ddrf::cuda::copy(ddrf::cuda::async, dev_p, p.ptr, stream, p.width, p.height);

                ddrf::cuda::synchronize_stream(stream);

                output_(output_type{std::move(dev_p), p.width, p.height, p.idx, p.phi, true, stream});
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
