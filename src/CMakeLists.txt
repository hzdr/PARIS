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

SET(COMMON_SOURCES  ddbvf.cpp
                    filesystem.cpp
                    geometry.cpp
                    his.cpp
                    main.cpp
                    preloader_stage.cpp
                    program_options.cpp
                    source_stage.cpp
                    task.cpp)

IF(DDAFA_ENABLE_CUDA)
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

    CUDA_ADD_EXECUTABLE(ddafa.cuda
                        cuda/device.cpp
                        cuda/error.cpp
                        cuda/memory.cpp
                        cuda/synchronization.cpp
                        filter_stage.cu
                        reconstruction_stage.cu
                        scheduler.cu
                        sink_stage.cu
                        weighting_stage.cu
                        ${COMMON_SOURCES})

    SET_PROPERTY(TARGET ddafa.cuda PROPERTY CXX_STANDARD 11)
    CUDA_ADD_CUFFT_TO_TARGET(ddafa.cuda)

    TARGET_COMPILE_DEFINITIONS(ddafa.cuda PRIVATE DDAFA_ENABLE_CUDA)

    TARGET_LINK_LIBRARIES(ddafa.cuda
                            ${Boost_LIBRARIES}
                            ${CMAKE_THREAD_LIBS_INIT})
ENDIF(DDAFA_ENABLE_CUDA)
