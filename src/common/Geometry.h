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

#include <cstdint>

namespace ddafa
{
	namespace common
	{
		struct Geometry
		{
			// Detector
			std::uint32_t det_pixels_row;			// number of pixels per row
			std::uint32_t det_pixels_column;			// number of pixels per column
			float det_pixel_size_horiz;				// size of pixel (distance between pixel centers) in horizontal direction
			float det_pixel_size_vert;				// size of pixel (distance between pixel centers) in vertical direction
			float det_offset_horiz;					// offset in horizontal direction
			float det_offset_vert;					// offset in vertical direction

			// Support
			float dist_src;							// distance between object and source
			float dist_det;							// distance between object and detector

			// Rotation
			float rot_angle;						// angle of rotation
		};
	}
}



#endif /* GEOMETRY_H_ */
