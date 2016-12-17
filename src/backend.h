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
 * Date: 25 November 2016
 * Authors: Jan Stephan
 */

#ifndef PARIS_BACKEND_H_
#define PARIS_BACKEND_H_

#if defined(PARIS_ENABLE_CUDA)
#include "cuda/backend.h"
#elif defined(PARIS_ENABLE_OPENCL)
#include "opencl/backend.h"
#elif defined(PARIS_ENABLE_OPENMP)
#include "openmp/backend.h"
#else
#include "generic/backend.h"
#endif

namespace paris
{
    #if defined(PARIS_ENABLE_CUDA)
    namespace backend = cuda;
    #elif defined(PARIS_ENABLE_OPENCL)
    namespace backend = opencl;
    #elif defined(PARIS_ENABLE_OPENMP)
    namespace backend = openmp;
    #else
    namespace backend = generic;
    #endif
}

#endif /* PARIS_BACKEND_H_ */
