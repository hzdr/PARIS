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

CUDA_ADD_EXECUTABLE(paris
                    ddbvf.cpp
                    filesystem.cpp
                    filter_stage.cu
                    geometry.cpp
                    his_loader.cpp
                    main.cpp
                    preloader_stage.cu
                    program_options.cpp
                    reconstruction_stage.cu
                    scheduler.cu
                    sink_stage.cu
                    source_stage.cpp
                    task.cpp
                    weighting_stage.cu)

SET_PROPERTY(TARGET paris PROPERTY CXX_STANDARD 11)
CUDA_ADD_CUFFT_TO_TARGET(paris)

TARGET_LINK_LIBRARIES(paris
                        ${Boost_LIBRARIES}
                        ${CMAKE_THREAD_LIBS_INIT})
