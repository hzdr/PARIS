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
 * Date: 19 August 2016
 * Authors: Jan Stephan
 */

#ifndef DDAFA_SINK_STAGE_H_
#define DDAFA_SINK_STAGE_H_

#include <functional>
#include <string>
#include <utility>

#include <ddrf/cuda/memory.h>

#include "metadata.h"

namespace ddafa
{
    class sink_stage
    {
        public:
            using input_type = std::pair<ddrf::cuda::pinned_host_ptr<float>, volume_metadata>;

        public:
            sink_stage(const std::string& path, const std::string& prefix);
            ~sink_stage();

            auto run() -> void;
            auto set_input_function(std::function<input_type(void)> input) noexcept -> void;

        private:
            std::function<input_type(void)> input_;
            int devices_;

            std::string path_;
            std::string prefix_;
    };
}



#endif /* DDAFA_SINK_STAGE_H_ */
