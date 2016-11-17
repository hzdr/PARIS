CUDA_ADD_EXECUTABLE(ddafa
                    d2h_stage.cpp
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
                    sink_stage_single.cpp
                    source_stage.cpp
                    task.cpp
                    tiff_saver.cpp
                    tiff_saver_single.cpp
                    version.cpp
                    weighting_stage.cu)

SET_PROPERTY(TARGET ddafa PROPERTY CXX_STANDARD 11)
CUDA_ADD_CUFFT_TO_TARGET(ddafa)

TARGET_LINK_LIBRARIES(ddafa
                        ${TIFF_LIBRARIES}
                        ${Boost_LIBRARIES}
                        ${CMAKE_THREAD_LIBS_INIT})
