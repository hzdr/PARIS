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

CUDA_ADD_EXECUTABLE(ddafa
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
                    tiff_saver.cpp
                    weighting_stage.cu)

SET_PROPERTY(TARGET ddafa PROPERTY CXX_STANDARD 11)
CUDA_ADD_CUFFT_TO_TARGET(ddafa)

TARGET_LINK_LIBRARIES(ddafa
                        ${TIFF_LIBRARIES}
                        ${Boost_LIBRARIES}
                        ${CMAKE_THREAD_LIBS_INIT})
