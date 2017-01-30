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

CMAKE_MINIMUM_REQUIRED(VERSION 3.5)
PROJECT(paris CXX)

IF(NOT DEFINED GLADOS_INCLUDE_PATH)
    MESSAGE(FATAL_ERROR "You must set GLADOS' include path!")
ENDIF(NOT DEFINED GLADOS_INCLUDE_PATH)

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules")

SET(Boost_USE_STATIC_LIBS OFF)
SET(Boost_USE_STATIC_RUNTIME OFF)
SET(Boost_USE_MULTITHREADED ON)
ADD_DEFINITIONS(-DBOOST_ALL_DYN_LINK=1)

FIND_PACKAGE(Boost REQUIRED COMPONENTS system filesystem log program_options)
FIND_PACKAGE(Threads REQUIRED)

FIND_PACKAGE(CUDA)
FIND_PACKAGE(OpenCL 1.2)
FIND_PACKAGE(OpenMP)

IF(CUDA_FOUND)
    SET(PARIS_ENABLE_CUDA TRUE)
ENDIF(CUDA_FOUND)

IF(OPENCL_FOUND)
    SET(PARIS_ENABLE_OPENCL TRUE)
ENDIF(OPENCL_FOUND)

IF(OPENMP_FOUND)
    SET(FFTW_USE_OPENMP ON)
    SET(FFTW_DOUBLE_PRECISION OFF)
    FIND_PACKAGE(FFTW)
    IF(FFTW_FOUND)
        INCLUDE_DIRECTORIES(${FFTW_INCLUDE_DIR})
        SET(PARIS_ENABLE_OPENMP TRUE)
    ELSE(FFTW_FOUND)
        MESSAGE(WARNING "FFTW not found - disabling OpenMP port")
    ENDIF(FFTW_FOUND)
ENDIF(OPENMP_FOUND)

INCLUDE_DIRECTORIES(${GLADOS_INCLUDE_PATH})

IF(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    SET(BASIC_WARNINGS "-Wall -Wextra -pedantic")
    SET(NUMERICAL_WARNINGS "-Wfloat-equal -Wconversion -Wnarrowing -Wsign-conversion")
    SET(DISABLED_WARNINGS "-Wno-missing-field-initializers")
    SET(CLASS_WARNINGS "-Wctor-dtor-privacy -Wdelete-non-virtual-dtor -Wnon-virtual-dtor -Woverloaded-virtual")
    SET(COMPAT_WARNINGS "-Wabi -Wvla")
    SET(BUG_WARNINGS "-Wnoexcept -Wswitch-enum -Wuninitialized -Wshadow -Wcast-qual -Wcast-align")

    SET(OPTIMIZATION_REPORTS "-fopt-info-vec-missed=vec_missed")
ENDIF(CMAKE_CXX_COMPILER_ID MATCHES "GNU")

IF(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    SET(BASIC_WARNINGS "-Wall -Wextra -pedantic")
    SET(NUMERICAL_WARNINGS "-Wfloat-equal -Wconversion -Wnarrowing -Wsign-conversion")
    SET(DISABLED_WARNINGS "-Wno-missing-field-initializers")
    SET(CLASS_WARNINGS "-Wnon-virtual-dtor -Woverloaded-virtual")
    SET(COMPAT_WARNINGS "-Wabi")
    SET(BUG_WARNINGS "-Wuninitialized -Wshadow -Wcast-qual")

    SET(OPTIMIZATION_REPORTS "-qopt-report -qopt-report-phase=vec")
ENDIF(CMAKE_CXX_COMPILER_ID MATCHES "Intel")

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${BASIC_WARNINGS} ${NUMERICAL_WARNINGS} ${CLASS_WARNINGS} ${COMPAT_WARNINGS} ${BUG_WARNINGS} ${DISABLED_WARNINGS}")
SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -ggdb -fno-omit-frame-pointer")
SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fno-omit-frame-pointer ${OPTIMIZATION_REPORTS}")

SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/Debug")
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/Release")

ADD_SUBDIRECTORY(src)
