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
 * Date: 25 November 2016
 * Authors: Jan Stephan
 */

#ifndef DDAFA_BACKEND_H_
#define DDAFA_BACKEND_H_

#if defined(DDAFA_ENABLE_CUDA)
#include "cuda/backend.h"
#elif defined(DDAFA_ENABLE_OPENCL)
#include "opencl/backend.h"
#elif defined(DDAFA_ENABLE_OPENMP)
#include "openmp/backend.h"
#else
#include "generic/backend.h"
#endif

namespace ddafa
{
    #if defined(DDAFA_ENABLE_CUDA)
    namespace backend = cuda;
    #elif defined(DDAFA_ENABLE_OPENCL)
    namespace backend = opencl;
    #elif defined(DDAFA_ENABLE_OPENMP)
    namespace backend = openmp;
    #else
    namespace backend = generic;
    #endif
}

#endif /* DDAFA_BACKEND_H_ */
