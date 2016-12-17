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
PROJECT(paris)

SET(Boost_USE_STATIC_LIBS OFF)
SET(Boost_USE_STATIC_RUNTIME OFF)
SET(Boost_USE_MULTITHREADED ON)
ADD_DEFINITIONS(-DBOOST_ALL_DYN_LINK=1)

IF(CMAKE_BUILD_TYPE MATCHES DEBUG)
    ADD_DEFINITIONS(-DPARIS_DEBUG=1)
ENDIF(CMAKE_BUILD_TYPE MATCHES DEBUG)

FIND_PACKAGE(Boost REQUIRED COMPONENTS system filesystem log program_options)
FIND_PACKAGE(CUDA REQUIRED)
FIND_PACKAGE(Threads REQUIRED)

IF(NOT DEFINED GLADOS_INCLUDE_PATH)
    MESSAGE(FATAL_ERROR "You must set ddrf's include path!")
ENDIF(NOT DEFINED GLADOS_INCLUDE_PATH)

INCLUDE_DIRECTORIES(${GLADOS_INCLUDE_PATH})

SET(CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -std=c++11;
    -gencode arch=compute_61,code=sm_61;
    -gencode arch=compute_35,code=sm_35;
    --default-stream per-thread;
    -prec-div=true;
    -prec-sqrt=true;
    -fmad=true)

SET(CUDA_NVCC_FLAGS_DEBUG
    ${CUDA_NVCC_FLAGS_DEBUG};
    -O0;
    -g;
    -G)

SET(CUDA_NVCC_FLAGS_RELEASE
    ${CUDA_NVCC_FLAGS_RELEASE};
    -O3;
    -lineinfo;
    -pg)

SET(ABI_OPTIONS "-fsigned-char -fsigned-bitfields -ffor-scope -fuse-cxa-atexit -fabi-version=0")
SET(BASIC_WARNINGS "-Wall -Wextra")
SET(NUMERICAL_WARNINGS "-Wfloat-equal -Wconversion -Wnarrowing -Wsign-conversion")
SET(DISABLED_WARNINGS "-Wno-missing-field-initializers")
SET(CLASS_WARNINGS "-Wctor-dtor-privacy -Wdelete-non-virtual-dtor -Wnon-virtual-dtor -Woverloaded-virtual")
SET(COMPAT_WARNINGS "-Wabi -Wvla")
SET(BUG_WARNINGS "-Wnoexcept -Wswitch-enum -Wuninitialized -Wshadow -Wcast-qual -Wcast-align")
SET(ADDITIONAL_WARNINGS "-Wuseless-cast")

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ABI_OPTIONS} ${BASIC_WARNINGS} ${NUMERICAL_WARNINGS} ${CLASS_WARNINGS} ${COMPAT_WARNINGS} ${BUG_WARNINGS} ${ADDITIONAL_WARNINGS} ${DISABLED_WARNINGS}")
SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -ggdb -fno-omit-frame-pointer")
SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fno-omit-frame-pointer")

SET(CUDA_PROPAGATE_HOST_FLAGS ON)
SET(CUDA_VERBOSE_BUILD OFF)
SET(CUDA_SEPARABLE_COMPILATION OFF)

SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/Debug")
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/Release")

ADD_SUBDIRECTORY(src)
