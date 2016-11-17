# ddafa - Dresden Accelerated Feldkamp Algorithm

## Build instructions

### Dependencies

* ddrf
* CUDA 7.5
* a CUDA 7.5 compatible C++ compiler with C++11 support (e.g. g++ 4.9)
* cuFFT 7.5
* libtiff >= 4.0.3
* Boost.System >= 1.54.0
* Boost.Log >= 1.54.0
* Boost.Program_options >= 1.54.0
* Boost.Thread >= 1.54.0

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
* --input /path/to/projection/directory
* --output /path/to/volume/directory

#### Optional parameters

* --help                        |            print help text and exit
* --geometry-format             |            prints required parameters in geometry file
* --angles  /path/to/angle.file |            override the rot_angle parameter in the geometry file; angles are loaded from the specified file on a per-projection basis
* --name volume_prefix          |            change the output volume's prefix (default: vol)

#### Geometry parameters

All parameters are specified as a "key=value" pair.

* det_pixels_row | [integer] number of pixels per detector row (= projection width)
* det_pixels_column | [integer] number of pixels per detector column (= projection height)
* det_pixel_size_horiz | [float] horizontal pixel size (= distance between pixel centers) in mm
* det_pixel_size_vert | [float] vertical pixel size (= distance between pixel centers) in mm
* det_offset_horiz | [float] horizontal detector offset in pixels
* det_offset_vert | [float] vertical detector offset in pixels
* dist_src | [float] distance between object (= center of rotation) and source in mm
* dist_det | [float] distance between object (= center of rotation) and detector in mm
* rot_angle | [float] angle step between two successive projections in °
