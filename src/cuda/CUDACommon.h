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
#include <utility>

#define BOOST_ALL_DYN_LINK
#include <boost/log/trivial.hpp>

#include "CUDAAssert.h"

namespace ddafa
{
	namespace impl
	{
		__device__ unsigned int getX();
		__device__ unsigned int getY();

		template <typename... Args>
		void launch1D(std::size_t input_size, void(*kernel)(Args...), Args... args)
		{
			// calculate max potential blocks
			int block_size;
			int min_grid_size;
			cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, kernel, 0, 0);

			// calculate de facto occupation based on input size
			int grid_size = (input_size + block_size - 1) / block_size;

			kernel<<<grid_size, block_size>>>(std::forward<Args>(args)...);
			assertCuda(cudaPeekAtLastError());

#ifdef DDAFA_DEBUG
			// calculate theoretical occupancy
			int max_active_blocks;
			cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks, kernel, block_size, 0);

			int device;
			cudaDeviceProp props;
			assertCuda(cudaGetDevice(&device));
			assertCuda(cudaGetDeviceProperties(&props, device));

			float occupancy = (max_active_blocks * block_size / props.warpSize) /
								float(props.maxThreadsPerMultiProcessor / props.warpSize);

			BOOST_LOG_TRIVIAL(debug) << "Launched blocks of size " << block_size << ". Theoretical occupancy: "
					<< occupancy;
#endif
		}

		template <typename... Args>
		void launch2D(std::size_t size_x, std::size_t size_y, void(*kernel)(Args...), Args... args)
		{
			auto roundUp = [](std::uint32_t numToRound, std::uint32_t multiple)
			{
				if(multiple == 0)
					return numToRound;

				int remainder = numToRound % multiple;
				if(remainder == 0)
					return numToRound;

				return numToRound + multiple - remainder;
			};

			int threads = roundUp(size_x * size_y, 1024);
			int blocks = threads / 1024;

			dim3 block_size(roundUp(size_x/blocks, 32), roundUp(size_y/blocks, 32));
			dim3 grid_size((size_x + block_size.x - 1)/block_size.x, (size_y + block_size.y - 1)/block_size.y);

#ifdef DDAFA_DEBUG
			BOOST_LOG_TRIVIAL(debug) << "Need " << blocks << " blocks";

			BOOST_LOG_TRIVIAL(debug) << "Grid size: " << grid_size.x << "x" << grid_size.y;
			BOOST_LOG_TRIVIAL(debug) << "Block size: " << block_size.x << "x" << block_size.y;
#endif
			kernel<<<grid_size, block_size>>>(args...);
			assertCuda(cudaPeekAtLastError());
		}
	}
}


#endif /* CUDACOMMON_H_ */
