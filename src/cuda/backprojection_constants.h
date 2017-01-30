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
 * Date: 04 December 2016
 * Authors: Jan Stephan <j.stephan@hzdr.de>
 */

#ifndef PARIS_BACKPROJECTION_CONSTANTS_H_
#define PARIS_BACKPROJECTION_CONSTANTS_H_

#include <cstdint>

namespace paris
{
        // constants for the current subvolume -- these never change between kernel executions
        struct backprojection_constants
        {
            std::uint32_t vol_dim_x;
            std::uint32_t vol_dim_x_full;
            std::uint32_t vol_dim_y;
            std::uint32_t vol_dim_y_full;
            std::uint32_t vol_dim_z;
            std::uint32_t vol_dim_z_full;
            std::uint32_t vol_offset;

            float l_vx_x;
            float l_vx_y;
            float l_vx_z;

            std::uint32_t proj_dim_x;
            std::uint32_t proj_dim_y;

            float l_px_x;
            float l_px_y;
            
            float delta_s;
            float delta_t;

            float d_so;
            float d_sd;
        };
}

#endif
