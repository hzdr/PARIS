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
 * Date: 19 August 2016
 * Authors: Jan Stephan <j.stephan@hzdr.de>
 */

#ifndef PARIS_SINK_H_
#define PARIS_SINK_H_

#include <string>

#include "backend.h"
#include "ddbvf.h"
#include "geometry.h"
#include "volume.h"

namespace paris
{
    class sink
    {
        public:
            sink(const std::string& path, const std::string& prefix, const volume_geometry& vol_geo);
            auto save(const backend::volume_device_type& v) -> void;

        private:
            std::string path_;
            std::string prefix_;
            ddbvf::handle_type handle_;

            volume_geometry vol_geo_;
    };
}

#endif /* PARIS_SINK_H_ */
