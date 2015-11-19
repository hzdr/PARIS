/*
 * Geometry.h
 *
 *  Created on: 19.11.2015
 *      Author: Jan Stephan
 *
 *      The Geometry structure contains the parameters needed for the backprojection.
 */

#ifndef GEOMETRY_H_
#define GEOMETRY_H_

#include <cstddef>
#include <cstdint>

struct Geometry
{
	// Detector
	std::size_t det_pixels_row;				// number of pixels per row
	std::size_t det_pixel_column;			// number of pixels per column
	float det_pixel_size_horiz;				// size of pixel (distance between pixel centers) in horizontal direction
	float det_pixel_size_vert;				// size of pixel (distance between pixel centers) in vertical direction

	// Support
	float dist_src;							// distance between source and object
	float dist_det;							// distance between source and detector

	// Target volume
	std::size_t vol_rows;					// number of rows
	std::size_t vol_columns;				// number of columns
	std::size_t vol_planes;					// number of planes
	float vol_voxel_width;
	float vol_voxel_height;
	float vol_voxel_depth;

	// Rotation
	float rot_angle;						// angle of rotation
};


#endif /* GEOMETRY_H_ */
