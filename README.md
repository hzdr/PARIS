# ddafa - Dresden Accelerated Feldkamp Algorithm

## Build instructions

### Dependencies

* CUDA 8.0 
* a CUDA 8.0 compatible C++ compiler with C++11 support (e.g. g++ 5.4)
* cuFFT 8.0
* Boost.System
* Boost.Log
* Boost.Program_options
* Boost.Thread

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
* --angles  /path/to/angle.file         | override the rot_angle parameter in the geometry file; angles are loaded from the specified file on a per-projection basis
* --name volume_prefix                  | change the output volume's prefix (default: vol)
* --roi                                 | enable region of interest support. Supply the ROI parameters with --roi-x1, --roi-x2, --roi-y1 and so on
* --quality [arg]                       | specify desired quality. --quality 1 (default) will consider every projection, --quality 2 every second and so on

#### Geometry parameters

All parameters are specified as a "key=value" pair.

* n_row | [integer] number of pixels per detector row (= projection width)
* n_col | [integer] number of pixels per detector column (= projection height)
* l_px_row | [float] horizontal pixel size (= distance between pixel centers) in mm
* l_px_col | [float] vertical pixel size (= distance between pixel centers) in mm
* delta_s | [float] horizontal detector offset in pixels
* delta_t | [float] vertical detector offset in pixels
* d_so | [float] distance between object (= center of rotation) and source in mm
* d_od | [float] distance between object (= center of rotation) and detector in mm
* delta_phi | [float] angle step between two successive projections in °
