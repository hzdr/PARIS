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
                        cuda/memory.cpp
                        ddbvf.cpp
                        filesystem.cpp
                        filter_stage.cu
                        geometry.cpp
                        his.cpp
                        main.cpp
                        preloader_stage.cu
                        program_options.cpp
                        reconstruction_stage.cu
                        scheduler.cu
                        sink_stage.cu
                        source_stage.cpp
                        task.cpp
                        weighting_stage.cu)

    SET_PROPERTY(TARGET ddafa.cuda PROPERTY CXX_STANDARD 11)
    CUDA_ADD_CUFFT_TO_TARGET(ddafa.cuda)

    TARGET_COMPILE_DEFINITIONS(ddafa.cuda PRIVATE DDAFA_ENABLE_CUDA)

    TARGET_LINK_LIBRARIES(ddafa.cuda
                            ${Boost_LIBRARIES}
                            ${CMAKE_THREAD_LIBS_INIT})
ENDIF(DDAFA_ENABLE_CUDA)
