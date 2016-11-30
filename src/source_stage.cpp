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

#include <algorithm>
#include <fstream>
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
    namespace
    {
        auto read_angles(const std::string& path) -> std::vector<float>
        {
            auto angles = std::vector<float>{};

            auto&& file = std::ifstream{path.c_str()};
            if(!file.is_open())
            {
                BOOST_LOG_TRIVIAL(warning) << "Could not open angle file at " << path << ", using default values.";
                return angles;
            }

            auto angle_string = std::string{};
            std::getline(file, angle_string);

            auto loc = std::locale{};
            if(angle_string.find(',') != std::string::npos)
                loc = std::locale{"de_DE.UTF-8"};

            file.seekg(0, std::ios_base::beg);
            file.imbue(loc);

            while(!file.eof())
            {
                auto angle = 0.f;
                file >> angle;
                angles.push_back(angle);
            }

            return angles;
        }
    }

    source_stage::source_stage() noexcept
    : output_{}
    {
    }

    auto source_stage::assign_task(task t) noexcept -> void
    {
        directory_ = t.input_path;

        enable_angles_ = t.enable_angles;
        angle_path_ = t.angle_path;

        quality_ = t.quality;
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

        auto angles = std::vector<float>{};
        if(enable_angles_)
            angles = read_angles(angle_path_);

        for(const auto& s : paths)
        {
            try
            {
                auto vec = loader.load(s);

                for(auto&& img : vec)
                {
                    if(i % quality_ == 0)
                    {
                        img.idx = i;

                        if(enable_angles_ && !angles.empty())
                            img.phi = angles.at(i);

                        output_(std::move(img));
                    }
                    ++i;
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
