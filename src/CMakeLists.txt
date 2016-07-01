CUDA_ADD_EXECUTABLE(ddafa
                    main.cpp
                    version.cpp
                    cuda/Feldkamp.cu
                    cuda/FeldkampScheduler.cpp
                    cuda/Filter.cu
                    cuda/Preloader.cu
                    cuda/Weighting.cu)

CUDA_ADD_CUFFT_TO_TARGET(ddafa)

TARGET_LINK_LIBRARIES(ddafa
                        ${DDRF_LIBRARY}
                        ${TIFF_LIBRARIES}
                        ${Boost_LIBRARIES}
                        ${CMAKE_THREAD_LIBS_INIT})