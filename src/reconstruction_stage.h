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
            reconstruction_stage(const geometry& det_geo, const volume_type& vol_geo, const std::vector<volume_type>& subvol_geos, bool predefined_angles);
            ~reconstruction_stage();
            reconstruction_stage(reconstruction_stage&& other) = default;
            auto operator=(reconstruction_stage&& other) -> reconstruction_stage& = default;

            auto run() -> void;
            auto set_input_function(std::function<input_type(void)> input) noexcept -> void;
            auto set_output_function(std::function<void(output_type)> output) noexcept -> void;

        private:
            auto safe_push(input_type) -> void;
            auto safe_pop(int) -> input_type;
            auto process(int) -> void;
            auto download_and_reset(int, std::uint32_t) -> void;

        private:
            std::function<input_type(void)> input_;
            std::function<void(output_type)> output_;

            geometry det_geo_;
            volume_type vol_geo_;
            bool predefined_angles_;
            output_type vol_out_;

            int devices_;

            std::vector<volume_type> subvol_vec_;
            using svv_size_type = typename decltype(subvol_vec_)::size_type;

            std::vector<volume_type> subvol_geo_vec_;
            using svgv_size_type = typename decltype(subvol_geo_vec_)::size_type;

            std::vector<std::queue<input_type>> input_vec_;
            using iv_size_type = typename decltype(input_vec_)::size_type;
    };
}



#endif /* DDAFA_RECONSTRUCTION_STAGE_H_ */
