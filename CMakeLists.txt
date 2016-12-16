# This file is part of the ddafa reconstruction program.
#
# Copyright (C) 2016 Helmholtz-Zentrum Dresden-Rossendorf
#
# ddafa is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ddafa is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ddafa. If not, see <http://www.gnu.org/licenses/>.

CMAKE_MINIMUM_REQUIRED(VERSION 3.5)
PROJECT(ddafa)

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
    SET(DDAFA_ENABLE_CUDA TRUE)
ENDIF(CUDA_FOUND)

IF(OPENCL_FOUND)
    SET(DDAFA_ENABLE_OPENCL TRUE)
ENDIF(OPENCL_FOUND)

IF(OPENMP_FOUND)
    SET(FFTW_USE_OPENMP ON)
    SET(FFTW_DOUBLE_PRECISION OFF)
    FIND_PACKAGE(FFTW REQUIRED)
    INCLUDE_DIRECTORIES(${FFTW_INCLUDE_DIR})
    SET(DDAFA_ENABLE_OPENMP TRUE)
ENDIF(OPENMP_FOUND)

INCLUDE_DIRECTORIES(${GLADOS_INCLUDE_PATH})

SET(BASIC_WARNINGS "-Wall -Wextra -pedantic")
SET(NUMERICAL_WARNINGS "-Wfloat-equal -Wconversion -Wnarrowing -Wsign-conversion")
SET(DISABLED_WARNINGS "-Wno-missing-field-initializers")
SET(CLASS_WARNINGS "-Wctor-dtor-privacy -Wdelete-non-virtual-dtor -Wnon-virtual-dtor -Woverloaded-virtual")
SET(COMPAT_WARNINGS "-Wabi -Wvla")
SET(BUG_WARNINGS "-Wnoexcept -Wswitch-enum -Wuninitialized -Wshadow -Wcast-qual -Wcast-align")

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${BASIC_WARNINGS} ${NUMERICAL_WARNINGS} ${CLASS_WARNINGS} ${COMPAT_WARNINGS} ${BUG_WARNINGS} ${DISABLED_WARNINGS}")
SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -ggdb -fno-omit-frame-pointer")
SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fno-omit-frame-pointer")

SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/Debug")
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/Release")

ADD_SUBDIRECTORY(src)
