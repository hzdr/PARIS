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

#ifndef PARIS_SOURCE_STAGE_H_
#define PARIS_SOURCE_STAGE_H_

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "his.h"
#include "task.h"

namespace paris
{
    class source_stage
    {
        public:
            using input_type = void;
            using output_type = his::image_type;

        public:
            source_stage() noexcept;
            auto assign_task(task t) noexcept -> void;
            auto run() -> void;
            auto set_output_function(std::function<void(output_type)> output) noexcept -> void;

        private:
            std::function<void(output_type)> output_;
            std::string directory_;
            
            bool enable_angles_;
            std::string angle_path_;

            std::uint16_t quality_;
    };
}

#endif /* PARIS_SOURCE_STAGE_H_ */
