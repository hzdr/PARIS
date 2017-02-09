/*
 * This file is part of the PARIS reconstruction program.
 *
 * Copyright (C) 2016 Helmholtz-Zentrum Dresden-Rossendorf
 *
 * PARIS is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PARIS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PARIS. If not, see <http://www.gnu.org/licenses/>.
 *
 * Date: 18 August 2016
 * Authors: Jan Stephan <j.stephan@hzdr.de>
 */

#include <algorithm>
#include <fstream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include <boost/log/trivial.hpp>

#include "backend.h"
#include "exception.h"
#include "filesystem.h"
#include "his.h"
#include "projection.h"
#include "source.h"

namespace paris
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

    source::source(const std::string& proj_dir,
                   bool enable_angles, const std::string& angle_file,
                   std::uint16_t quality) noexcept
    : drained_{true}, enable_angles_{enable_angles}, quality_{quality}
    {
        paths_ = read_directory(proj_dir);
        if(!paths_.empty())
            drained_ = false;

        if(enable_angles_)
            angles_ = read_angles(angle_file);
    }

    auto source::load_next() -> output_type
    {
        thread_local static auto i = 0u;
        if(queue_.empty())
        {
            auto done = false;
            while(!done)
            {
                auto path = std::move(paths_[0u]);
                paths_.erase(std::begin(paths_));

                if(i % quality_ != 0)
                {
                    ++i;
                    continue;
                }

                auto vec = his::load(path);
                if(vec.empty())
                {
                    BOOST_LOG_TRIVIAL(warning) << "Skipping invalid file at " << paths_[0u];
                    --i;
                    continue;
                }

                for(auto&& p : vec)
                {
                    if(i % quality_ != 0)
                    {
                        ++i;
                        continue;
                    }

                    p.idx = i;

                    if(enable_angles_ && !angles_.empty())
                        p.phi = angles_[i];

                    queue_.push(std::move(p));
                    ++i;
                }

                done = !queue_.empty();
            }
        }

        auto p = std::move(queue_.front());
        queue_.pop();

        if((paths_.size() < quality_) && queue_.empty())
            drained_ = true;

        return p;
    }

    auto source::drained() const noexcept -> bool
    {
        return drained_;
    }
}
