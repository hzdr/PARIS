/*
 * CUDAFeldkamp.h
 *
 *  Created on: 12.11.2015
 *      Author: Jan Stephan
 *
 *      This class is the concrete backprojection implementation for the Stage class.
 */

#ifndef CUDAFELDKAMP_H_
#define CUDAFELDKAMP_H_

#include <atomic>
#include <deque>
#include <future>
#include <map>
#include <thread>
#include <type_traits>
#include <vector>

#include <ddrf/Image.h>
#include <ddrf/cuda/HostMemoryManager.h>
#include <ddrf/cuda/Image.h>
#include <ddrf/cuda/Memory.h>
#include <ddrf/default/Image.h>
#include <ddrf/observer/Observer.h>

#include "../common/Geometry.h"

#include "FeldkampScheduler.h"

namespace ddafa
{
	namespace cuda
	{
		class Feldkamp
		{
			public:
				using input_type = ddrf::Image<ddrf::cuda::Image<float>>;
				using output_type = ddrf::Image<ddrf::def::Image<float, ddrf::cuda::HostMemoryManager<float>>>;
				using volume_type = ddrf::cuda::pitched_device_ptr<float, ddrf::cuda::sync_copy_policy, std::true_type>;

			public:
				Feldkamp(const common::Geometry&);
				auto process(input_type&&) -> void;
				auto wait() -> output_type;
				auto set_input_num(std::uint32_t) noexcept -> void;

			private:
				auto create_volumes(int) -> void;

			protected:
				~Feldkamp() = default;

			private:
				int devices_;
				FeldkampScheduler<float> scheduler_;
				common::Geometry geo_;
				std::uint32_t input_num_;
				std::atomic_bool input_num_set_;
				std::map<int, std::vector<volume_type>> volume_map_;
				std::map<int, std::deque<std::future<bool>>> reconstruction_futures_;
		};
	}
}


#endif /* CUDAFELDKAMP_H_ */
