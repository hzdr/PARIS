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

#include <cufft.h>

namespace ddafa
{
	namespace impl
	{
		inline void assertCuda(cudaError_t err)
		{
			if(err != cudaSuccess)
				throw std::runtime_error("CUDA assertion failed: " + std::string(cudaGetErrorString(err)));
		}

		inline std::string getCufftErrorString(cufftResult result)
		{
			switch(result)
			{
				case CUFFT_SUCCESS: return "The cuFFT operation was successful";
				case CUFFT_INVALID_PLAN: return "cuFFT was passed an invalid plan handle";
				case CUFFT_ALLOC_FAILED: return "cuFFT failed to allocate GPU or CPU memory";
				case CUFFT_INVALID_TYPE: return "Invalid type";
				case CUFFT_INVALID_VALUE: return "Invalid pointer or parameter";
				case CUFFT_INTERNAL_ERROR: return "Driver or internal cuFFT library error";
				case CUFFT_EXEC_FAILED: return "Failed to execute an FFT on the GPU";
				case CUFFT_SETUP_FAILED: return "The cuFFT library failed to initialize";
				case CUFFT_INVALID_SIZE: return "User specified an invalid transform size";
				case CUFFT_UNALIGNED_DATA: return "Unaligned data";
				case CUFFT_INCOMPLETE_PARAMETER_LIST: return "Missing parameters in call";
				case CUFFT_INVALID_DEVICE: return "Execution of plan was on different GPU than plan creation";
				case CUFFT_PARSE_ERROR: return "Internal plan database error";
				case CUFFT_NO_WORKSPACE: return "No workspace has been provided prior to plan execution";
				case CUFFT_NOT_IMPLEMENTED: return "This feature is not implemented for your cuFFT version";
				case CUFFT_LICENSE_ERROR: return "NVIDIA license required. The file was either not found, is out of data, or otherwise invalid";
				default: return "Unknown error";
			}
		}

		inline void assertCufft(cufftResult result)
		{
			if(result != CUFFT_SUCCESS)
				throw std::runtime_error("cuFFT assertion failed: " + getCufftErrorString(result));
		}

		template <typename... Args>
		void launch1D(std::size_t input_size, void(*kernel)(Args...), Args... args)
		{
			// calculate max potential blocks
			int block_size;
			int min_grid_size;
			cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, kernel, 0, 0);

			// calculate de facto occupation based on input size
			int grid_size = (input_size + block_size - 1) / block_size;

			kernel<<<grid_size, block_size>>>(args...);
			assertCuda(cudaPeekAtLastError());
			assertCuda(cudaStreamSynchronize(0));

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

			std::cout << "Launched blocks of size " << block_size << ". Theoretical occupancy: "
					<< occupancy << std::endl;
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
			std::cout << "Need " << blocks << " blocks" << std::endl;

			std::cout << "Grid size: " << grid_size.x << "x" << grid_size.y << std::endl;
			std::cout << "Block size: " << block_size.x << "x" << block_size.y << std::endl;
#endif

			kernel<<<grid_size, block_size>>>(args...);
			assertCuda(cudaPeekAtLastError());
			assertCuda(cudaStreamSynchronize(0));
		}
	}
}


#endif /* CUDACOMMON_H_ */
