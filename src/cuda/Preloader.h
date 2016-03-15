#ifndef CUDA_PRELOADER_H_
#define CUDA_PRELOADER_H_

#include <array>
#include <cstddef>
#include <map>
#include <thread>
#include <vector>

#include <ddrf/Image.h>
#include <ddrf/Queue.h>
#include <ddrf/cuda/Image.h>
#include <ddrf/cuda/HostMemoryManager.h>
#include <ddrf/default/Image.h>

#include "../common/Geometry.h"

#include "FeldkampScheduler.h"

namespace ddafa
{
	namespace cuda
	{
		class Preloader
		{
			public:
				using input_type = ddrf::Image<ddrf::def::Image<float, ddrf::cuda::HostMemoryManager<float>>>;
				using output_img_type = ddrf::Image<ddrf::cuda::Image<float>>;
				using output_type = std::array<ddrf::Image<ddrf::cuda::Image<float>>, 4>;

			public:
				Preloader(const common::Geometry& geo);
				auto process(input_type&& input) -> void;
				auto wait() -> output_type;

			protected:
				~Preloader() = default;

			private:
				auto sort() -> void;
				auto split() -> void;
				auto distribute() -> void;
				auto finish() -> void;

				auto uploadAndSend(int device) -> void;

			private:
				ddrf::Queue<output_type> results_;
				std::vector<input_type> input_buf_;
				std::vector<std::array<input_type, 4>> sorted_input_;
				std::map<int, std::map<std::size_t, std::vector<std::array<input_type, 4>>>> device_chunk_map_;
				/*
				 * device_cunk_map_ is the central element of the preloader. The outer map (std::map<int, ...>) connects the inner map
				 * to its corresponding device.
				 * The inner map (std::map<std::size_t, ...>) contains the individual chunks. Its key is the id of the current chunk, its value
				 * is the vector that contains the sorted projection blocks.
				 *
				 * Example: We are using two devices. Each device processes two chunks, e.g. device #0 the first two chunks, device #1 the last
				 * two. Thus, device_chunk_map_[0] returns the chunks for the first device and device_chunk_map_[1] the chunks for the second device.
				 *
				 * The chunks are processed serially; device #0 starts with chunk #0 and device #1 with chunk #2.
				 *
				 * The enclosed vector contains the blocks for each chunk. At chunk #0 the first array contains the projection data of projections
				 * at {0°, 90°, 180°, 270°}, the second array contains {0.25°, 90.25°, 180.25°, 270.25°} etc.
				 *
				 * Each chunk contains a subset of the full projection data, e.g., chunk #0 contains the rows 0 - 255, chunk #1 the rows 256 - 511 etc.
				 */

				std::vector<std::thread> distribution_threads_;
				FeldkampScheduler<float> scheduler_;


				/*
				 * Ablauf:
				 *
				 * Geometrie an Scheduler [implementiert]
				 * Scheduler berechnet Speicherverbrauch auf dem Device [implementiert]
				 * Scheduler berechnet Höhe der Teilvolumen [implementiert]
				 * Scheduler berechnet Höhe der Projektionen [implementiert]
				 * Scheduler ordnet Teilvolumen und -projektionen den Devices zu [implementiert]
				 * Feldkampstufe allokiert Teilvolumen auf den Devices [TODO]
				 * Preloader lädt alle Projektionen [implementiert]
				 * Preloader sortiert Projektionen [implementiert]
				 * Preloader fragt die Höhe der Teilprojektionen ab [implementiert]
				 * Preloader spaltet Projektionen in Teilprojektionen auf [implementiert]
				 * Preloader lädt Teilprojektionen auf zugehörige Devices [implementiert]
				 * Preloader schickt Device-Teilprojektionen an die nächste Stufe weiter [implementiert]
				 */
		};
	}
}

#endif /* CUDA_PRELOADER_H_ */
