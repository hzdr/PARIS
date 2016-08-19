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
