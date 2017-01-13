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

SET(COMMON_SOURCES  ddbvf.cpp
                    filesystem.cpp
                    filter_stage.cpp
                    geometry.cpp
                    his.cpp
                    main.cpp
                    preloader_stage.cpp
                    program_options.cpp
                    reconstruction_stage.cpp
                    sink_stage.cpp
                    source_stage.cpp
                    task.cpp
                    weighting_stage.cpp)

IF(PARIS_ENABLE_CUDA)
    SET(CUDA_NVCC_FLAGS
        ${CUDA_NVCC_FLAGS};
        -std=c++11;
        -gencode arch=compute_30,code=sm_30;
        --default-stream per-thread;
        -prec-div=true;
        -prec-sqrt=true;
        -fmad=true;
        )

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

    SET(CUDA_PROPAGATE_HOST_FLAGS ON)
    SET(CUDA_VERBOSE_BUILD OFF)
    SET(CUDA_SEPARABLE_COMPILATION OFF)

    CUDA_ADD_EXECUTABLE(paris.cuda
                        cuda/device.cpp
                        cuda/error.cpp
                        cuda/filtering.cpp
                        cuda/filter_kernels.cu
                        cuda/reconstruction_kernels.cu
                        cuda/subvolume_information.cpp
                        cuda/synchronization.cpp
                        cuda/weighting_kernels.cu
                        ${COMMON_SOURCES})

    SET_PROPERTY(TARGET paris.cuda PROPERTY CXX_STANDARD 11)
    CUDA_ADD_CUFFT_TO_TARGET(paris.cuda)

    TARGET_COMPILE_DEFINITIONS(paris.cuda PRIVATE PARIS_ENABLE_CUDA)

    TARGET_LINK_LIBRARIES(paris.cuda
                            ${Boost_LIBRARIES}
                            ${CMAKE_THREAD_LIBS_INIT})
ENDIF(PARIS_ENABLE_CUDA)

IF(PARIS_ENABLE_OPENMP)
    ADD_EXECUTABLE(paris.openmp
                   openmp/filtering.cpp
                   openmp/reconstruction.cpp
                   openmp/subvolume_information.cpp
                   openmp/weighting.cpp
                   ${COMMON_SOURCES})

    SET_PROPERTY(TARGET paris.openmp PROPERTY CXX_STANDARD 14)
    TARGET_COMPILE_DEFINITIONS(paris.openmp PRIVATE PARIS_ENABLE_OPENMP)
    TARGET_COMPILE_OPTIONS(paris.openmp PRIVATE ${OpenMP_CXX_FLAGS})

    TARGET_LINK_LIBRARIES(paris.openmp
                            ${OpenMP_CXX_FLAGS}
                            ${Boost_LIBRARIES}
                            ${FFTW_LIBRARIES}
                            ${CMAKE_THREAD_LIBS_INIT})
ENDIF(PARIS_ENABLE_OPENMP)
