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

#include <atomic>
#include <functional>
#include <queue>
#include <utility>
#include <vector>

#include <ddrf/cuda/memory.h>
#include <ddrf/memory.h>

#include "geometry.h"
#include "metadata.h"

namespace ddafa
{
    class reconstruction_stage
    {
        private:
            using device_allocator = ddrf::cuda::device_allocator<float, ddrf::memory_layout::pointer_2D>;
            using pool_allocator = ddrf::pool_allocator<float, ddrf::memory_layout::pointer_2D, device_allocator>;
            using smart_pointer = typename pool_allocator::smart_pointer;
            using volume_type = std::pair<ddrf::cuda::pitched_device_ptr<float>, volume_metadata>;

        public:
            using input_type = std::pair<smart_pointer, projection_metadata>;
            using output_type = std::pair<ddrf::cuda::pinned_host_ptr<float>, volume_metadata>;

        public:
            reconstruction_stage(const geometry& det_geo, const volume_metadata& vol_geo, const std::vector<volume_metadata>& subvol_geos, bool predefined_angles);
            reconstruction_stage(reconstruction_stage&& other) noexcept;
            ~reconstruction_stage();
            auto operator=(reconstruction_stage&& other) noexcept -> reconstruction_stage&;

            auto run() -> void;
            auto set_input_function(std::function<input_type(void)> input) noexcept -> void;
            auto set_output_function(std::function<void(output_type)> output) noexcept -> void;

        private:
            auto safe_push(input_type) -> void;
            auto safe_pop(int) -> input_type;
            auto process(int) -> void;
            auto download_and_reset(int, volume_metadata) -> void;

        private:
            std::function<input_type(void)> input_;
            std::function<void(output_type)> output_;

            geometry det_geo_;
            volume_metadata vol_geo_;
            bool predefined_angles_;
            output_type vol_out_;

            int devices_;
            std::vector<volume_type> subvol_vec_;
            std::vector<std::vector<volume_metadata>> subvol_geo_vec_;
            std::vector<std::queue<input_type>> input_vec_;
            std::atomic_flag lock_ = ATOMIC_FLAG_INIT;
    };
}



#endif /* DDAFA_RECONSTRUCTION_STAGE_H_ */
