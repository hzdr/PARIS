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

#ifndef DDAFA_TIFF_SAVER_SINGLE_H_
#define DDAFA_TIFF_SAVER_SINGLE_H_

#include <string>
#include <utility>

#include <ddrf/cuda/memory.h>

#include "metadata.h"

namespace ddafa
{
    class tiff_saver_single
    {
        public:
            tiff_saver_single() noexcept = default;
            auto save(std::pair<ddrf::cuda::pinned_host_ptr<float>, projection_metadata> vol, const std::string& path) const -> void;
    };
}



#endif /* DDAFA_TIFF_SAVER_H_ */
