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
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <boost/log/trivial.hpp>

#include "exception.h"
#include "filesystem.h"
#include "source_stage.h"
#include "task.h"


namespace ddafa
{
    source_stage::source_stage() noexcept
    : output_{}
    {
    }

    auto source_stage::assign_task(task t) noexcept -> void
    {
        directory_ = t.input_path;
    }

    auto source_stage::run() -> void
    {
        auto loader = his_loader{};
        auto i = 0u;

        auto paths = std::vector<std::string>{};

        try
        {
            paths = read_directory(directory_);
        }
        catch(const std::runtime_error& e)
        {
            BOOST_LOG_TRIVIAL(fatal) << "source_stage::source_stage() failed to obtain file paths: " << e.what();
            throw stage_construction_error{"source_stage::source_stage() failed"};
        }

        for(const auto& s : paths)
        {
            try
            {
                auto vec = loader.load(s);

                for(auto&& img : vec)
                {
                    img.idx = i;
                    ++i;
                    output_(std::move(img));
                }
            }
            catch(const std::system_error& e)
            {
                BOOST_LOG_TRIVIAL(fatal) << "source_stage::run(): Could not open file at " << s << " : " << e.what();
                throw stage_runtime_error{"source_stage::run() failed"};
            }
            catch(const std::runtime_error& e)
            {
                BOOST_LOG_TRIVIAL(warning) << "source_stage::run(): Skipping invalid HIS file at " << s << " : " << e.what();
                continue;
            }
        }

        // all frames loaded, send empty image
        output_(output_type{});
        BOOST_LOG_TRIVIAL(info) << "All projections loaded.";
    }

    auto source_stage::set_output_function(std::function<void(output_type)> output) noexcept -> void
    {
        output_ = output;
    }
}
