# PARIS - Portable and Accelerated 3D Reconstruction tool for radiation based Imaging Systems

## Build instructions

### Branch selection

We strongly recommend the usage of the latest stable branch (currently 0.2). Issue the following command to switch to
the stable branch (do the same in ddrf's repository as well!):

```
git checkout 0.2
```

### Dependencies

* CMake 3.5
* CUDA 8.0 
* a CUDA 8.0 compatible C++ compiler with C++11 support (e.g. g++ 5.4)
* cuFFT 8.0
* Boost.System
* Boost.Log
* Boost.Program\_options
* Boost.Thread

### Preparations

Before building you need to ensure that the generated GPU code matches your target architecture. Edit the top-level
CMakeLists.txt and find the following lines:

```
-gencode arch=compute_61,code=sm_61;
-gencode arch=compute_35,code=sm_35;
```

Most users can delete the second line as they typically won't utilize two different GPUs. The correct values can be
looked up [here](https://developer.nvidia.com/cuda-gpus) (just leave out the dot). For example, an adaption for the
GeForce 940M looks like this:

```
-gencode arch=compute_50,code=sm_50;
```

### Building

You have to clone the ddrf repository first. Then execute the following steps in the ddafa directory:

```
mkdir build
cd build
cmake -DDDRF_INCLUDE_PATH=/path/to/ddrf/include -DCMAKE_BUILD_TYPE=Release ..
make
```
## Execution

### Usage

#### Required parameters

* --geometry /path/to/file.geo

#### Optional parameters

* --help                                | print help text and exit
* --geometry-format                     | prints required parameters in geometry file
* --input /path/to/projection/directory | path to projections. If this parameter is supplied, "--output" must be specified as well
* --output /path/to/volume/directory    | target directory. If this parameter is supplied, "--input" must be specified as well
* --angles  /path/to/angle.file         | override the rot\_angle parameter in the geometry file; angles are loaded from the specified file on a per-projection basis
* --name volume\_prefix                 | change the output volume's prefix (default: vol)
* --roi                                 | enable region of interest support. Supply the ROI parameters with --roi-x1, --roi-x2, --roi-y1 and so on
* --quality [arg]                       | specify desired quality. --quality 1 (default) will consider every projection, --quality 2 every second and so on

#### Geometry parameters

All parameters are specified as a "key=value" pair.

* n\_row | [integer] number of pixels per detector row (= projection width)
* n\_col | [integer] number of pixels per detector column (= projection height)
* l\_px\_row | [float] horizontal pixel size (= distance between pixel centers) in mm
* l\_px\_col | [float] vertical pixel size (= distance between pixel centers) in mm
* delta\_s | [float] horizontal detector offset in pixels
* delta\_t | [float] vertical detector offset in pixels
* d\_so | [float] distance between object (= center of rotation) and source in mm
* d\_od | [float] distance between object (= center of rotation) and detector in mm
* delta\_phi | [float] angle step between two successive projections in °

## File format

ddrf's file format description can be found in the [wiki](https://github.com/HZDR-FWDF/ddafa/wiki/ddbvf-format).
