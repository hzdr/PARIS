/*
 * CUDACommon.h
 *
 *  Created on: 24.11.2015
 *      Author: Jan Stephan
 *
 *      Various helper functions for CUDA
 */

#ifndef CUDACOMMON_H_
#define CUDACOMMON_H_

#include <cstddef>
#ifdef DDAFA_DEBUG
#include <iostream>
#endif

namespace ddafa
{
	namespace impl
	{
		template <typename... Args>
		void launch(std::size_t input_size, void(*kernel)(Args...), Args... args)
		{
			// calculate max potential blocks
			int block_size;
			int min_grid_size;
			cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, kernel, 0, 0);

			// calculate de facto occupation based on input size
			int grid_size = (input_size + block_size - 1) / block_size;

			kernel<<<grid_size, block_size>>>(args...);
			cudaDeviceSynchronize();

#ifdef DDAFA_DEBUG
			// calculate theoretical occupancy
			int max_active_blocks;
			cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks, kernel, block_size, 0);

			int device;
			cudaDeviceProp props;
			cudaGetDevice(&device);
			cudaGetDeviceProperties(&props, device);

			float occupancy = (max_active_blocks * block_size / props.warpSize) /
								float(props.maxThreadsPerMultiProcessor / props.warpSize);

			std::cout << "Launched blocks of size " << block_size << ". Theoretical occupancy: "
					<< occupancy << std::endl;
#endif
		}
	}
}


#endif /* CUDACOMMON_H_ */
