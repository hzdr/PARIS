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

#ifndef DDAFA_HIS_LOADER_H_
#define DDAFA_HIS_LOADER_H_

#include <string>
#include <utility>
#include <vector>

#include <ddrf/cuda/memory.h>
#include <ddrf/memory.h>

#include "metadata.h"

namespace ddafa
{
    class his_loader
    {
        public:
            using cuda_host_allocator = ddrf::cuda::host_allocator<float, ddrf::memory_layout::pointer_2D>;
            using pool_allocator = ddrf::pool_allocator<float, ddrf::memory_layout::pointer_2D, cuda_host_allocator>;
            using smart_pointer = typename pool_allocator::smart_pointer;
            using image_type = std::pair<smart_pointer, projection_metadata>;

            his_loader() noexcept = default;
            auto load(const std::string& path) -> std::vector<image_type>;

        private:
             pool_allocator alloc_;
    };
}



#endif /* DDAFA_HIS_LOADER_H_ */
