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
 * Date: 18 August 2016
 * Authors: Jan Stephan <j.stephan@hzdr.de>
 */

#ifndef PARIS_EXCEPTION_H_
#define PARIS_EXCEPTION_H_

#include <exception>
#include <stdexcept>

namespace paris
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

#endif /* PARIS_EXCEPTION_H_ */
