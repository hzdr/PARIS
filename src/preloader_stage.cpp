/*
 * This file is part of the ddafa reconstruction program.
 *
 * Copyright (C) 2016 Helmholtz-Zentrum Dresden-Rossendorf
 *
 * ddafa is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ddafa is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with ddafa. If not, see <http://www.gnu.org/licenses/>.
 *
 * Date: 18 August 2016
 * Authors: Jan Stephan <j.stephan@hzdr.de>
 */

#include <cstddef>
#include <cstdlib>
#include <functional>
#include <utility>

#include <boost/log/trivial.hpp>

#include "backend.h"
#include "exception.h"
#include "projection.h"
#include "preloader_stage.h"
#include "task.h"

namespace ddafa
{
    preloader_stage::preloader_stage(std::size_t pool_limit, const backend::device_handle& device) noexcept
    : input_{}, output_{}, device_{device}, limit_{pool_limit}, pool_{pool_limit}
    {}

    preloader_stage::~preloader_stage()
    {
        auto err = backend::set_device_noexcept(device_);
        if(err != backend::success)
        {
            backend::print_error("preloader_stage::~preloader_stage() encountered error: ", err);
            std::exit(EXIT_FAILURE);
        }

        pool_.release();
    }

    preloader_stage::preloader_stage(const preloader_stage& other)
    : input_{other.input_}, output_{other.output_}, device_{other.device_} 
    , limit_{other.limit_}, pool_{other.limit_}
    {
    }

    auto preloader_stage::operator=(const preloader_stage& other) -> preloader_stage&
    {
        input_ = other.input_;
        output_ = other.output_;
        device_ = other.device_;
        limit_ = other.limit_;
        pool_.release();
        pool_ = pool_allocator{limit_};
        return *this;
    }

    auto preloader_stage::assign_task(task) noexcept -> void
    {}

    auto preloader_stage::run() -> void
    {
        auto sre = stage_runtime_error{"preloader_stage::run() failed"};
        try
        {
            backend::set_device(device_);
            while(true)
            {
                auto p = input_();
                if(!p.valid)
                    break;

                auto dev_p = pool_.allocate_smart(p.width, p.height);
                auto handle = backend::make_async_handle();

                // initialize the destination data before copying
                backend::fill(backend::async, dev_p, 0, handle, p.width, p.height);
                backend::copy(backend::async, dev_p, p.ptr, handle, p.width, p.height);

                backend::synchronize(handle);

                output_(output_type{std::move(dev_p), p.width, p.height, p.idx, p.phi, true, handle});
            }

            // Uploaded all projections to the GPU, notify the next stage that we are done here
            output_(output_type{});
            BOOST_LOG_TRIVIAL(info) << "Uploaded all projections to the device(s)";
        }
        catch(const backend::bad_alloc& ba)
        {
            BOOST_LOG_TRIVIAL(fatal) << "preloader_stage::run() encountered a bad_alloc: " << ba.what();
            throw sre;
        }
        catch(const backend::invalid_argument& ia)
        {
            BOOST_LOG_TRIVIAL(fatal) << "preloader_stage::run() passed an invalid argument to the CUDA runtime: " << ia.what();
            throw sre;
        }
        catch(const backend::runtime_error& re)
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
