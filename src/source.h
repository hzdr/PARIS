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

#ifndef PARIS_SOURCE_H_
#define PARIS_SOURCE_H_

#include <cstdint>
#include <string>
#include <queue>
#include <vector>

#include "backend.h"
#include "projection.h"

namespace paris
{
    class source
    {
        private:
            using output_type = backend::projection_host_type;

        public:
            source(const std::string& proj_dir,
                   bool enable_angles = false,
                   const std::string& angle_file = "",
                   std::uint16_t quality = 1) noexcept;

            auto load_next() -> output_type;
            auto drained() const noexcept -> bool;

        private:
            std::vector<std::string> paths_;
            std::queue<output_type> queue_;
            bool drained_;
            bool used_poison_;
            
            bool enable_angles_;
            std::vector<float> angles_;
            std::uint16_t quality_;
    };
}

#endif /* PARIS_SOURCE_H_ */
