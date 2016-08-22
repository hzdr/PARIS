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
#include "metadata.h"
#include "source_stage.h"


namespace ddafa
{
    source_stage::source_stage(const std::string& dir)
    : loader_{}, output_{}
    {
        try
        {
            paths_ = read_directory(dir);
        }
        catch(const std::runtime_error& e)
        {
            BOOST_LOG_TRIVIAL(fatal) << "source_stage::source_stage() failed to obtain file paths: " << e.what();
            throw stage_construction_error{"source_stage::source_stage() failed"};
        }
    }

    auto source_stage::run() -> void
    {
        auto i = 0u;
        for(const auto& s : paths_)
        {
            auto vec = std::vector<output_type>{};

            try
            {
                vec = loader_.load(s);
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

            for(auto&& img : vec)
            {
                img.second.index = i;
                ++i;
                output_(std::move(img));
            }
        }

        // all frames loaded, send empty image
        output_(std::make_pair(nullptr, projection_metadata{0, 0, 0, 0.f, false, 0}));
    }

    auto source_stage::set_output_function(std::function<void(output_type)> output) noexcept -> void
    {
        output_ = output;
    }
}
