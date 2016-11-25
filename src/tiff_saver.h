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
 * Date: 18 August 2016
 * Authors: Jan Stephan
 */

#ifndef DDAFA_TIFF_SAVER_H_
#define DDAFA_TIFF_SAVER_H_

#include <string>
#include <utility>

#include <ddrf/cuda/memory.h>

#include "volume.h"

namespace ddafa
{
    class tiff_saver
    {
        public:
            tiff_saver() noexcept = default;
            auto save(volume<ddrf::cuda::pinned_host_ptr<float>> vol, const std::string& path) const -> void;
    };
}



#endif /* DDAFA_TIFF_SAVER_H_ */
