CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12)
PROJECT(ddafa)

SET(Boost_USE_STATIC_LIBS OFF)
SET(Boost_USE_STATIC_RUNTIME OFF)
SET(Boost_USE_MULTITHREADED ON)
ADD_DEFINITIONS(-DBOOST_ALL_DYN_LINK=1)

FIND_PACKAGE(Boost 1.54.0 REQUIRED COMPONENTS system log program_options thread)
FIND_PACKAGE(CUDA REQUIRED)
FIND_PACKAGE(TIFF REQUIRED)
FIND_PACKAGE(Threads REQUIRED)

IF(NOT DEFINED DDRF_INCLUDE_PATH)
    MESSAGE(FATAL_ERROR "You must set ddrf's include path!")
ENDIF(NOT DEFINED DDRF_INCLUDE_PATH)

INCLUDE_DIRECTORIES(${DDRF_INCLUDE_PATH})

IF(NOT DEFINED DDRF_LIBRARY)
    MESSAGE(FATAL_ERROR "You must set the path to libddrf.so!")
ENDIF(NOT DEFINED DDRF_LIBRARY)

SET(CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -std=c++11;
    -gencode arch=compute_20,code=sm_20;
    -gencode arch=compute_20,code=sm_21;
    -gencode arch=compute_30,code=sm_30;
    -gencode arch=compute_32,code=sm_32;
    -gencode arch=compute_35,code=sm_35;
    -gencode arch=compute_50,code=sm_50;
    -gencode arch=compute_52,code=sm_52;
    -gencode arch=compute_53,code=sm_53;
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

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 ${ABI_OPTIONS} ${BASIC_WARNINGS} ${NUMERICAL_WARNINGS} ${CLASS_WARNINGS} ${COMPAT_WARNINGS} ${BUG_WARNINGS} ${ADDITIONAL_WARNINGS} ${DISABLED_WARNINGS}")
SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -ggdb -fno-omit-framepointer")
SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fno-omit-frame-pointer")

SET(CUDA_PROPAGATE_HOST_FLAGS ON)
SET(CUDA_VERBOSE_BUILD OFF)
SET(CUDA_SEPARABLE_COMPILATION ON)

SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/Debug")
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/Release")

ADD_SUBDIRECTORY(src)