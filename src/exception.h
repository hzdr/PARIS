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

#ifndef DDAFA_EXCEPTION_H_
#define DDAFA_EXCEPTION_H_

#include <exception>
#include <stdexcept>

namespace ddafa
{
    class stage_construction_error : public std::runtime_error
    {
        public:
            using std::runtime_error::runtime_error;
    };

    class stage_runtime_error : public std::runtime_error
    {
        public:
            using std::runtime_error::runtime_error;
    };
}

#endif /* DDAFA_EXCEPTION_H_ */
