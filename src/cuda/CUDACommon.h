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
		__device__ auto getX() -> unsigned int;
		__device__ auto getY() -> unsigned int;

		template <typename... Args>
		auto launch1D(std::size_t input_size, void(*kernel)(Args...), Args... args) -> void
		{
			// calculate max potential blocks
			auto block_size = int{};
			auto min_grid_size = int{};
			cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, kernel, 0, 0);

			// calculate de facto occupation based on input size
			auto grid_size = (input_size + block_size - 1) / block_size;

			kernel<<<grid_size, block_size>>>(std::forward<Args>(args)...);
			assertCuda(cudaPeekAtLastError());

#ifdef DDAFA_DEBUG
			// calculate theoretical occupancy
			auto max_active_blocks = int{};
			cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks, kernel, block_size, 0);

			auto device = int{};
			cudaDeviceProp props;
			assertCuda(cudaGetDevice(&device));
			assertCuda(cudaGetDeviceProperties(&props, device));

			auto occupancy = (max_active_blocks * block_size / props.warpSize) /
								float(props.maxThreadsPerMultiProcessor / props.warpSize);

			BOOST_LOG_TRIVIAL(debug) << "Launched blocks of size " << block_size << ". Theoretical occupancy: "
					<< occupancy;
#endif
		}

		template <typename... Args>
		auto launch2D(std::size_t size_x, std::size_t size_y, void(*kernel)(Args...), Args... args) -> void
		{
			auto roundUp = [](std::uint32_t numToRound, std::uint32_t multiple)
			{
				if(multiple == 0)
					return numToRound;

				auto remainder = numToRound % multiple;
				if(remainder == 0)
					return numToRound;

				return numToRound + multiple - remainder;
			};

			auto threads = roundUp(static_cast<unsigned int>(size_x * size_y), 1024);
			auto blocks = threads / 1024;

			auto block_size = dim3{roundUp(static_cast<unsigned int>(size_x/blocks), 32),
									roundUp(static_cast<unsigned int>(size_y)/blocks, 32)};
			auto grid_size = dim3{static_cast<unsigned int>((size_x + block_size.x - 1)/block_size.x),
									static_cast<unsigned int>((size_y + block_size.y - 1)/block_size.y)};

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
