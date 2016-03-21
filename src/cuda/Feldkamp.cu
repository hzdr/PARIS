#include <array>
#include <cstddef>
#include <cmath>
#include <future>
#include <thread>
#include <utility>
#include <vector>

#define BOOST_ALL_DYN_LINK
#include <boost/log/trivial.hpp>

#include <ddrf/Image.h>
#include <ddrf/cuda/Check.h>
#include <ddrf/cuda/Coordinates.h>
#include <ddrf/cuda/Launch.h>

#include "../common/Geometry.h"
#include "Feldkamp.h"
#include "FeldkampScheduler.h"

namespace ddafa
{
	namespace cuda
	{
		__global__ void init_volume(float* vol, std::size_t width, std::size_t height, std::size_t depth, std::size_t pitch)
		{
			auto x = ddrf::cuda::getX();
			auto y = ddrf::cuda::getY();
			auto z = ddrf::cuda::getZ();

			if((x < width) && (y < height) && (z < depth))
			{
				auto slice_pitch = pitch * height;
				auto slice = reinterpret_cast<char*>(vol) + z * slice_pitch;
				auto row = reinterpret_cast<float*>(slice + y * pitch);

				row[x] = 0.f;
			}
		}

		__device__ auto float_coordinate(unsigned int coord, std::size_t dim) -> float
		{
			auto center = static_cast<float>(dim)/2.f;
			return static_cast<float>(coord) - center;
		}

		__device__ auto int_coordinate(float coord, std::size_t dim) -> unsigned int
		{
			auto center = static_cast<float>(dim)/2.f;
			return static_cast<unsigned int>(roundf(coord + center));
		}

		__global__ void backproject(float *vol, std::size_t vol_w, std::size_t vol_h, std::size_t vol_d, std::size_t vol_pitch,
									const float *proj, std::size_t proj_w, std::size_t proj_h, std::size_t proj_pitch,
									unsigned int i, float angle, float dist_src, float dist_det,
									std::uint32_t num_proj)
		{
			auto x = ddrf::cuda::getX();
			auto y = ddrf::cuda::getY();
			auto z = ddrf::cuda::getZ();

			if((x < vol_w) && (y < vol_h) && (z < vol_d))
			{
				auto slice_pitch = vol_pitch * vol_h;
				auto slice = reinterpret_cast<char*>(vol) + z * slice_pitch;
				auto row = reinterpret_cast<float*>(slice + y * vol_pitch);

				auto x_f = float_coordinate(x, vol_w);
				auto y_f = float_coordinate(y, vol_h);
				auto z_f = float_coordinate(z, vol_d);

				auto denominator = dist_src - x_f * cosf(angle) - y_f * sinf(angle);
				auto dist_so = dist_src + dist_det;
				auto u = (dist_so * (-x_f * sinf(angle) + y_f * cosf(angle))) / denominator;
				auto v = (dist_so * z_f) / denominator;
				auto w2 = dist_src / denominator;

				auto u_i = int_coordinate(u, proj_w);
				auto v_i = int_coordinate(v, proj_h);

				auto proj_row = reinterpret_cast<const float*>(reinterpret_cast<const char*>(proj) + v_i * proj_pitch);

				row[x] += (1.f / (2.f * M_PI * num_proj)) * proj_row[u_i] * w2;
			}
		}


		Feldkamp::Feldkamp(const common::Geometry& geo)
		: scheduler_{FeldkampScheduler<float>::instance(geo)}, done_{false}
		, geo_(geo), input_num_{0u}, input_num_set_{false}, current_img_{0u}, current_angle_{0.f}
		, output_{geo_.det_pixels_row, geo_.det_pixels_row, geo_.det_pixels_column}
		{
			ddrf::cuda::check(cudaGetDeviceCount(&devices_));
			std::vector<std::thread> creation_threads;
			for(auto i = 0; i < devices_; ++i)
			{
				creation_threads.emplace_back(&Feldkamp::create_volumes, this, i);
				auto pr = std::promise<bool>{};
				processor_futures_[i].emplace_back(pr.get_future());
				pr.set_value(true);
			}

			for(auto&& t : creation_threads)
				t.join();
		}

		auto Feldkamp::process(input_type&& img) -> void
		{
			if(!img.valid())
			{
				finish();
				return;
			}

			auto pr = std::promise<bool>{};
			processor_futures_[img.device()].emplace_back(pr.get_future());
			processor_threads_.emplace_back(&Feldkamp::processor, this, std::move(img), std::move(pr));
		}

		auto Feldkamp::wait() -> output_type
		{
			while(!done_)
				std::this_thread::yield();

			return std::move(output_);
		}

		auto Feldkamp::processor(input_type&& img, std::promise<bool> pr) -> void
		{
			auto device = img.device();
			auto future = std::move(processor_futures_[device].front());
			processor_futures_[device].pop_front();
			auto start = future.get();
			start = !start;

			ddrf::cuda::check(cudaSetDevice(device));
			BOOST_LOG_TRIVIAL(debug) << "cuda::Feldkamp: Processing on device #" << device;

			while(!input_num_set_)
				std::this_thread::yield();

			auto volumes = volume_map_[device];
			for(auto& v : volumes) // FIXME: Fix this after testing
			{
				ddrf::cuda::launch(v.width(), v.height(), v.depth(),
									backproject,
									v.data(), v.width(), v.height(), v.depth(), v.pitch(),
									static_cast<const float*>(img.data()), img.width(), img.height(), img.pitch(), current_img_, current_angle_,
									geo_.dist_src, geo_.dist_det, input_num_);
			}

			++current_img_;
			current_angle_ += 0.25f;
			pr.set_value(true);
		}

		auto Feldkamp::set_input_num(std::uint32_t num) noexcept -> void
		{
			input_num_ = num;
			input_num_set_ = true;
		}

		auto Feldkamp::create_volumes(int device) -> void
		{
			ddrf::cuda::check(cudaSetDevice(device));
			auto volume_num = scheduler_.chunkNumber(device);
			auto dimensions = scheduler_.chunkDimensions(device);
			for(auto i = 0u; i < volume_num; ++i)
			{
				auto first_row = dimensions[i].first;
				auto last_row = dimensions[i].second;
				auto rows = last_row - first_row + 1u;

				auto ptr = ddrf::cuda::make_device_ptr<float>(geo_.det_pixels_row, geo_.det_pixels_row, rows);
				ddrf::cuda::launch(ptr.width(), ptr.height(), ptr.depth(),
									init_volume,
									ptr.get(), ptr.width(), ptr.height(), ptr.depth(), ptr.pitch());

				volume_map_[device].emplace_back(ptr.width(), ptr.height(), ptr.depth(), std::move(ptr));
			}
		}

		auto Feldkamp::finish() -> void
		{
			BOOST_LOG_TRIVIAL(debug) << "cuda::Feldkamp: Received poisonous pill, called finish()";

			for(auto&& t : processor_threads_)
				t.join();

			for(auto& kv : processor_futures_)
			{
				for(auto& f : kv.second)
				{
					auto val = f.get();
					val = !val;
				}
			}

			merge_volumes();
			done_ = true;
		}

		auto Feldkamp::merge_volumes() -> void
		{
			for(auto i = 0; i < devices_; ++i)
			{
				ddrf::cuda::check(cudaSetDevice(i));
				auto dimensions = scheduler_.chunkDimensions(i);
				for(auto& v : volume_map_[i])
				{
					auto first_row = dimensions[i].first;
					auto output_start = output_.data() + first_row * output_.width() * output_.height();

					auto parms = cudaMemcpy3DParms{0};
					auto uchar_width = output_.width() * sizeof(float)/sizeof(unsigned char);
					auto height = output_.height();
					parms.srcPtr = make_cudaPitchedPtr(reinterpret_cast<unsigned char*>(v.data()), v.pitch(), uchar_width, height);
					parms.dstPtr = make_cudaPitchedPtr(reinterpret_cast<unsigned char*>(output_start), output_.pitch(), uchar_width, height);
					parms.extent = make_cudaExtent(uchar_width, height, output_.depth());
					parms.kind = cudaMemcpyDeviceToHost;
					ddrf::cuda::check(cudaMemcpy3D(&parms));
				}
			}
		}
	}
}
