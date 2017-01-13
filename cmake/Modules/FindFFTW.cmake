# This file is part of the PARIS reconstruction program.
#
# Copyright (C) 2016 Helmholtz-Zentrum Dresden-Rossendorf
#
# PARIS is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PARIS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PARIS. If not, see <http://www.gnu.org/licenses/>.

# - Find FFTW
# Find FFTW include directories and libraries
#
# Use this module by invoking FIND_PACKAGE with the form:
#
#   FIND_PACKAGE(FFTW [REQUIRED])
#
# This module finds headers and the fitting library. Results
# are reported in the following variables:
#
#   FFTW_FOUND
#   FFTW_INCLUDE_DIR
#   FFTW_LIBRARY            - the FFTW library with the specified precision (see
#                             below)
#   FFTW_MT_LIBRARY         - the multithreaded FFTW library (needs to be passed
#                             to the linker BEFORE FFTW_LIBRARY). Included in
#                             FFTW_LIBRARIES if FFTW_USE_MULTITHREADED is set
#   FFTW_OMP_LIBRARY        - the OpenMP FFTW library (needs to be passed to the
#                             linker BEFORE FFTW_LIBRARY). Included in
#                             FFTW_LIBRARIES if FFTW_USE_OPENMP is set
#   FFTW_LIBRARIES          - the FFTW library and additional libraries in the right
#                             order
#
# FFTW libraries come in multiple variants encoded in their file name.
# Users or projects may tell this module to which variant to find by
# setting variables:
#
#   FFTW_USE_MULTITHREADED  - Set to OFF to use the non-multithreaded
#                             libraries ('_threads' tag). Default is ON.
#   FFTW_USE_OPENMP         - Set to ON to use the OpenMP libraries ('_omp' tag).
#                             Default is OFF. Automatically disables
#                             FFTW_USE_MULTIRHREADED.
#   FFTW_DOUBLE_PRECISION   - Set to OFF to use the single precision libraries
#                             ('f' tag). Default is ON.
#   FFTW_LONG_PRECISION     - Set to ON to use the long double precision
#                             libraries ('l' tag). Default is OFF. Automatically
#                             disables FFTW_DOUBLE_PRECISION

IF(NOT DEFINED FFTW_USE_MULTITHREADED)
    SET(FFTW_USE_MULTITHREADED ON)
ENDIF(NOT DEFINED FFTW_USE_MULTITHREADED)

IF(FFTW_USE_OPENMP)
    SET(FFTW_USE_MULTITHREADED OFF)
ENDIF(FFTW_USE_OPENMP)

IF(NOT DEFINED FFTW_DOUBLE_PRECISION)
    SET(FFTW_DOUBLE_PRECISION ON)
ENDIF(NOT DEFINED FFTW_DOUBLE_PRECISION)

IF(FFTW_LONG_PRECISION)
    SET(FFTW_DOUBLE_PRECISION OFF)
ENDIF(FFTW_LONG_PRECISION)

IF(FFTW_INCLUDE_DIR)
    # FFTW already found, don't look again
    SET(FFTW_FIND_QUIETLY TRUE)
ENDIF(FFTW_INCLUDE_DIR)

FIND_PATH(FFTW_INCLUDE_DIR fftw3.h)

# Standard case - find double precision libraries
IF(FFTW_DOUBLE_PRECISION)
    # we always need the basic library
    FIND_LIBRARY(FFTW_LIBRARY fftw3)

    IF(FFTW_USE_MULTITHREADED)
        FIND_LIBRARY(FFTW_MT_LIBRARY fftw3_threads)
    ENDIF(FFTW_USE_MULTITHREADED)

    IF(FFTW_USE_OPENMP)
        FIND_LIBRARY(FFTW_OMP_LIBRARY fftw3_omp)
    ENDIF(FFTW_USE_OPENMP)
ENDIF(FFTW_DOUBLE_PRECISION)

# Find single precision libraries
IF(NOT FFTW_DOUBLE_PRECISION AND NOT FFTW_LONG_PRECISION)
    #we always need the basic library
    FIND_LIBRARY(FFTW_LIBRARY fftw3f)

    IF(FFTW_USE_MULTITHREADED)
        FIND_LIBRARY(FFTW_MT_LIBRARY fftw3f_threads)
    ENDIF(FFTW_USE_MULTITHREADED)

    IF(FFTW_USE_OPENMP)
        FIND_LIBRARY(FFTW_OMP_LIBRARY fftw3f_omp)
    ENDIF(FFTW_USE_OPENMP)
ENDIF(NOT FFTW_DOUBLE_PRECISION AND NOT FFTW_LONG_PRECISION)

# Find long double precision libraries
IF(FFTW_LONG_PRECISION)
    # we always need the basic library
    FIND_LIBRARY(FFTW_LIBRARY fftw3l)

    IF(FFTW_USE_MULTITHREADED)
        FIND_LIBRARY(FFTW_MT_LIBRARY fftw3l_threads)
    ENDIF(FFTW_USE_MULTITHREADED)

    IF(FFTW_USE_OPENMP)
        FIND_LIBRARY(FFTW_OMP_LIBRARY fftw3l_omp)
    ENDIF(FFTW_USE_OPENMP)
ENDIF(FFTW_LONG_PRECISION)

# Build FFTW_LIBRARIES
SET(FFTW_LIBRARIES ${FFTW_MT_LIBRARY} ${FFTW_OMP_LIBRARY} ${FFTW_LIBRARY})

MESSAGE(STATUS "Found FFTW libraries: ${FFTW_LIBRARIES}")

# handle REQUIRED and QUIET parameters
FIND_PACKAGE_HANDLE_STANDARD_ARGS(FFTW DEFAULT_MSG FFTW_INCLUDE_DIR FFTW_LIBRARY)

MARK_AS_ADVANCED(FFTW_LIBRARIES FFTW_INCLUDE_DIR)
