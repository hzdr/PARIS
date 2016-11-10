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

#ifndef DDAFA_RECONSTRUCTION_STAGE_H_
#define DDAFA_RECONSTRUCTION_STAGE_H_

#include <cstdint>
#include <functional>
#include <queue>
#include <utility>
#include <vector>

#include <ddrf/cuda/memory.h>
#include <ddrf/memory.h>

#include "geometry.h"
#include "projection.h"
#include "region_of_interest.h"
#include "scheduler.h"
#include "task.h"
#include "volume.h"

namespace ddafa
{
    class reconstruction_stage
    {
        private:
            using device_allocator = ddrf::cuda::device_allocator<float, ddrf::memory_layout::pointer_2D>;
            using pool_allocator = ddrf::pool_allocator<float, ddrf::memory_layout::pointer_2D, device_allocator>;
            using smart_pointer = typename pool_allocator::smart_pointer;

        public:
            using input_type = projection<smart_pointer>;
            using output_type = volume<ddrf::cuda::pinned_host_ptr<float>>;
            using volume_type = volume<ddrf::cuda::pitched_device_ptr<float>>;

        public:
            reconstruction_stage(int device) noexcept;
            ~reconstruction_stage() = default;
            reconstruction_stage(reconstruction_stage&& other) = default;
            auto operator=(reconstruction_stage&& other) -> reconstruction_stage& = default;

            auto assign_task(task t) noexcept -> void;
            auto run() -> void;
            auto set_input_function(std::function<input_type(void)> input) noexcept -> void;
            auto set_output_function(std::function<void(output_type)> output) noexcept -> void;

        private:
            auto process(int) -> void;
            auto download_and_reset(int, std::uint32_t) -> void;

        private:
            std::function<input_type(void)> input_;
            std::function<void(output_type)> output_;

            detector_geometry det_geo_;
            volume_geometry vol_geo_;
            subvolume_geometry subvol_geo_;

            bool enable_angles_;

            bool enable_roi_;
            region_of_interest roi_;

            int device_;

            std::uint32_t task_id_;
            std::uint32_t task_num_;
    };
}



#endif /* DDAFA_RECONSTRUCTION_STAGE_H_ */
