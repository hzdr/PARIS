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

#ifndef DDAFA_SOURCE_STAGE_H_
#define DDAFA_SOURCE_STAGE_H_

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "his_loader.h"
#include "task.h"

namespace ddafa
{
    class source_stage
    {
        public:
            using input_type = void;
            using output_type = typename his_loader::image_type;

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



#endif /* DDAFA_SOURCE_STAGE_H_ */
