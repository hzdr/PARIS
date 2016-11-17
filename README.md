# ddafa - Dresden Accelerated Feldkamp Algorithm

## Build instructions

### Dependencies

* CUDA 8.0 
* a CUDA 8.0 compatible C++ compiler with C++11 support (e.g. g++ 5.4)
* cuFFT 8.0
* libtiff
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
* --input /path/to/projection/directory
* --output /path/to/volume/directory

#### Optional parameters

* --help                        |            print help text and exit
* --geometry-format             |            prints required parameters in geometry file
* --angles  /path/to/angle.file |            override the rot_angle parameter in the geometry file; angles are loaded from the specified file on a per-projection basis
* --name volume_prefix          |            change the output volume's prefix (default: vol)
* --roi                         |            enable region of interest support. Supply the ROI parameters with --roi-x1, --roi-x2, --roi-y1 and so on

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
