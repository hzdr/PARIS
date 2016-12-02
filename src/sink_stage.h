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
 * Date: 19 August 2016
 * Authors: Jan Stephan <j.stephan@hzdr.de>
 */

#ifndef DDAFA_SINK_STAGE_H_
#define DDAFA_SINK_STAGE_H_

#include <cstddef>
#include <functional>
#include <string>
#include <utility>

#include <ddrf/cuda/memory.h>

#include "geometry.h"
#include "task.h"
#include "volume.h"

namespace ddafa
{
    class sink_stage
    {
        public:
            using input_type = volume<ddrf::cuda::pinned_host_ptr<float>>;
            using output_type = void;

        public:
            sink_stage(const std::string& path, const std::string& prefix, const volume_geometry& vol_geo, std::size_t tasks);

            auto assign_task(task t) noexcept -> void;
            auto run() -> void;
            auto set_input_function(std::function<input_type(void)> input) noexcept -> void;

        private:
            std::function<input_type(void)> input_;

            std::string path_;
            std::string prefix_;

            std::size_t tasks_;

            volume_geometry vol_geo_;
    };
}



#endif /* DDAFA_SINK_STAGE_H_ */
