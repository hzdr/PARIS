CUDA_ADD_EXECUTABLE(ddafa
                    device_to_host_stage.cpp
                    filesystem.cpp
                    filter_stage.cu
                    geometry_calculator.cpp
                    his_loader.cpp
                    main.cpp
                    preloader_stage.cu
                    reconstruction_stage.cu
                    sink_stage.cu
                    sink_stage_single.cu
                    source_stage.cpp
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
