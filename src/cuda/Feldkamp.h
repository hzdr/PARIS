#ifndef CUDA_FELDKAMP_H_
#define CUDA_FELDKAMP_H_

#include <atomic>
#include <deque>
#include <future>
#include <map>
#include <thread>
#include <type_traits>
#include <vector>

#include <ddrf/Image.h>
#include <ddrf/Queue.h>
#include <ddrf/Volume.h>
#include <ddrf/cuda/DeviceMemoryManager.h>
#include <ddrf/cuda/HostMemoryManager.h>
#include <ddrf/cuda/Memory.h>

#include "../common/Geometry.h"

#include "FeldkampScheduler.h"

namespace ddafa
{
	namespace cuda
	{
		class Feldkamp
		{
			public:
				using input_type = ddrf::Image<ddrf::cuda::DeviceMemoryManager<float>>;
				using output_type = ddrf::Volume<ddrf::cuda::HostMemoryManager<float>>;
				using volume_type = ddrf::Volume<ddrf::cuda::DeviceMemoryManager<float>>;

			public:
				Feldkamp(const common::Geometry&);
				auto process(input_type&&) -> void;
				auto wait() -> output_type;
				auto set_input_num(std::uint32_t) noexcept -> void;

			private:
				auto create_volumes(int) -> void;
				auto processor(input_type&&, std::promise<bool>) -> void;
				auto finish() -> void;
				auto merge_volumes() -> void;

			protected:
				~Feldkamp() = default;

			private:
				ddrf::Queue<output_type> results_;
				int devices_;
				bool done_;

				FeldkampScheduler<float> scheduler_;
				common::Geometry geo_;

				std::uint32_t input_num_;
				std::atomic_bool input_num_set_;

				std::uint32_t current_img_;
				float current_angle_;

				std::map<int, std::vector<volume_type>> volume_map_;

				std::vector<std::thread> processor_threads_;
				std::map<int, std::deque<std::future<bool>>> processor_futures_;
		};
	}
}


#endif /* CUDA_FELDKAMP_H_ */
