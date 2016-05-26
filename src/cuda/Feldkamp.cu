#include <algorithm>
#include <array>
#include <cstddef>
#include <cmath>
#include <fstream>
#include <locale>
#include <mutex>
#include <thread>
#include <string>
#include <utility>
#include <vector>

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
		inline __device__ auto vol_centered_coordinate(unsigned int coord, std::size_t dim, float size) -> float
		{
			auto size2 = size / 2.f;
			return -(dim * size2) + size2 + coord * size;
		}

		inline __device__ auto proj_centered_coordinate(unsigned int coord, std::size_t dim, float size, float offset) -> float
		{
			auto size2 = size / 2.f;
			auto min = -(dim * size2) - offset;
			return size2 + ((static_cast<float>(coord) + 1.f / 2.f)) * size + min;
		}

		// round and cast as needed
		inline __device__ auto proj_real_coordinate(float coord, std::size_t dim, float size, float offset) -> float
		{
			auto size2 = size / 2.f;
			auto min = -(dim * size2) - offset;
			return (coord - min) / size - (1.f / 2.f);
		}

		template <class T>
		inline __device__ auto as_unsigned(T x) -> unsigned int
		{
			return static_cast<unsigned int>(x);
		}

		__device__ auto interpolate(float h, volatile float v, const float* proj, std::size_t proj_width, std::size_t proj_height, std::size_t proj_pitch,
									std::size_t proj_offset, std::size_t proj_height_full, float pixel_size_x, float pixel_size_y,
									float offset_x, float offset_y)
		-> float
		{
			auto h_real = proj_real_coordinate(h, proj_width, pixel_size_x, offset_x);
			auto v_real = proj_real_coordinate(v, proj_height_full, pixel_size_y, offset_y);

			auto h_j0 = floorf(h_real);
			auto h_j1 = h_j0 + 1.f;
			auto v_i0 = floorf(v_real) - static_cast<float>(proj_offset);
			auto v_i1 = v_i0 + 1.f;

			auto w_h0 = h_real - h_j0;
			auto w_v0 = v_real - v_i0 - static_cast<float>(proj_offset);

			auto w_h1 = 1.f - w_h0;
			auto w_v1 = 1.f - w_v0;

			auto h_j0_ui = as_unsigned(h_j0);
			auto h_j1_ui = as_unsigned(h_j1);
			auto v_i0_ui = as_unsigned(v_i0);
			auto v_i1_ui = as_unsigned(v_i1);

			// ui coordinates might be invalid due to negative v_i0, thus
			// bounds checking
			auto h_j0_valid = (h_j0 >= 0.f);
			auto h_j1_valid = (h_j1 < static_cast<float>(proj_width));
			auto v_i0_valid = (v_i0 >= 0.f);
			auto v_i1_valid = (v_i1 < static_cast<float>(proj_height));

			auto upper_row = reinterpret_cast<const float*>(reinterpret_cast<const char*>(proj) + v_i0_ui * proj_pitch);
			auto lower_row = reinterpret_cast<const float*>(reinterpret_cast<const char*>(proj) + v_i1_ui * proj_pitch);

			auto tl = 0.f;
			auto bl = 0.f;
			auto tr = 0.f;
			auto br = 0.f;
			if(h_j0_valid && h_j1_valid && v_i0_valid && v_i1_valid)
			{
				tl = upper_row[h_j0_ui];
				bl = lower_row[h_j0_ui];
				tr = upper_row[h_j1_ui];
				br = lower_row[h_j1_ui];
			}

			auto val = 	w_h1	* w_v1	* tl +
						w_h1	* w_v0	* bl +
						w_h0	* w_v1	* tr +
						w_h0	* w_v0	* br;

			return val;
		}

		__global__ void backproject(float* __restrict__ vol, std::size_t vol_w, std::size_t vol_h, std::size_t vol_d, std::size_t vol_pitch,
									std::size_t vol_offset, std::size_t vol_d_full, float voxel_size_x, float voxel_size_y, float voxel_size_z,
									const float* __restrict__ proj, std::size_t proj_w, std::size_t proj_h, std::size_t proj_pitch,
									std::size_t proj_offset, std::size_t proj_h_full, float pixel_size_x, float pixel_size_y,
									float pixel_offset_x, float pixel_offset_y, float angle_sin, float angle_cos, float dist_src, float dist_sd)
		{
			auto k = ddrf::cuda::getX();
			auto l = ddrf::cuda::getY();
			auto m = ddrf::cuda::getZ();

			if((k < vol_w) && (l < vol_h) && (m < vol_d))
			{
				auto slice_pitch = vol_pitch * vol_h;
				auto slice = reinterpret_cast<char*>(vol) + m * slice_pitch;
				auto row = reinterpret_cast<float*>(slice + l * vol_pitch);

				// add offset for the current subvolume
				auto m_off = m + vol_offset;

				// get centered coordinates -- volume center is at (0, 0, 0) and the top slice is at -(vol_d_off / 2)
				auto x_k = vol_centered_coordinate(k, vol_w, voxel_size_x);
				auto y_l = vol_centered_coordinate(l, vol_h, voxel_size_y);
				auto z_m = vol_centered_coordinate(m_off, vol_d_full, voxel_size_z);

				// rotate coordinates
				auto s = x_k * angle_cos + y_l * angle_sin;
				auto t = -x_k * angle_sin + y_l * angle_cos;
				auto z = z_m;

				// project rotated coordinates
				auto factor = dist_sd / (s + dist_src);
				auto h = t * factor;
				auto v = z * factor;

				// get projection value by interpolation
				auto det = interpolate(h, v, proj, proj_w, proj_h, proj_pitch, proj_offset, proj_h_full, pixel_size_x, pixel_size_y,
										pixel_offset_x, pixel_offset_y);

				// backproject
				auto u = -(dist_src / (s + dist_src));
				row[k] += 0.5f * det * powf(u, 2.f);
			}
		}

		Feldkamp::Feldkamp(const common::Geometry& geo, const std::string& angles)
		: done_{false}, geo_(geo), dist_sd_{geo_.dist_det + geo_.dist_src}
		, vol_geo_(FeldkampScheduler::instance(geo, cuda::volume_type::single_float).get_volume_geometry())
		, input_num_{0u}, input_num_set_{false}, current_img_{0u}, current_angle_{0.f}
		, output_{vol_geo_.dim_x, vol_geo_.dim_y, vol_geo_.dim_z}
		{
			if(!angles.empty())
				parse_angles(angles);

			CHECK(cudaGetDeviceCount(&devices_));
			std::vector<std::thread> creation_threads;

			for(auto i = 0; i < devices_; ++i)
			{
				creation_threads.emplace_back(&Feldkamp::create_volume, this, i);
				processor_threads_[i] = std::thread{&Feldkamp::processor, this, i};
			}

			for(auto&& t : creation_threads)
				t.join();
		}

		Feldkamp::~Feldkamp()
		{
			// this is the last stage in the pipeline, time to clean up CUDA
			cudaDeviceReset();
		}

		auto Feldkamp::parse_angles(const std::string& angles) -> void
		{
			auto&& file = std::ifstream{angles.c_str()};
			if(!file.is_open())
			{
				BOOST_LOG_TRIVIAL(warning) << "cuda::Feldkamp: Could not open angle file at " << angles << ", using default values.";
				return;
			}

			auto angle_string = std::string{};
			std::getline(file, angle_string);

			auto loc = std::locale{};
			if(angle_string.find(',') != std::string::npos)
				loc = std::locale{"de_DE.UTF-8"};

			file.seekg(0, std::ios_base::beg);
			file.imbue(loc);

			while(file.tellg() != std::ios_base::end)
			{
				auto angle = 0.f;
				file >> angle;
				sin_tab_.emplace_back(std::sin(angle * M_PI / 180.f));
				cos_tab_.emplace_back(std::cos(angle * M_PI / 180.f));
			}

			angle_tabs_created_ = true;
		}

		auto Feldkamp::process(input_type&& img) -> void
		{
			if(img.valid())
				map_imgs_[img.device()].push(std::move(img));
			else
			{
				BOOST_LOG_TRIVIAL(debug) << "cuda::Feldkamp: Received poisonous pill, finishing...";
				for(auto i = 0; i < devices_; ++i)
					map_imgs_[i].push(input_type());

				for(auto i = 0; i < devices_; ++i)
					processor_threads_[i].join();

				results_.push(std::move(output_));
				results_.push(output_type());
				done_ = true;
				BOOST_LOG_TRIVIAL(info) << "cuda::Feldkamp: Done.";
			}
		}

		auto Feldkamp::wait() -> output_type
		{
			while(!done_)
				std::this_thread::yield();

			return results_.take();
		}

		auto Feldkamp::processor(int device) -> void
		{
			CHECK(cudaSetDevice(device));

			auto proj_count = 0u;
			auto vol_count = 0u;

			while(true)
			{
				auto img = map_imgs_[device].take();
				if(!img.valid())
				{
					// our work here is done
					download_and_reset_volume(device, vol_count);
					break;
				}

				++proj_count;
				if(input_num_set_ && (proj_count >= input_num_))
				{
					// we are processing the next subvolume -> download the old subvolume to the host and reset the GPU volume to 0
					download_and_reset_volume(device, vol_count);
					proj_count = 1u;
					++vol_count;
				}

				BOOST_LOG_TRIVIAL(debug) << "cuda::Feldkamp: Processing image #" << img.index() << " on device #" << device;

				if(img.index() % 10 == 0)
					BOOST_LOG_TRIVIAL(info) << "cuda::Feldkamp: Device #" << device << " is processing image #" << img.index() << " of volume #" << vol_count;

				auto& v = volume_map_[device];

				// the geometry offsets are measured in pixels
				auto& scheduler = FeldkampScheduler::instance(geo_, cuda::volume_type::single_float);
				auto vol_offset = scheduler.get_volume_offset(device, vol_count);
				auto proj_offset = scheduler.get_subproj_offset(device, vol_count);
				auto offset_horiz = geo_.det_offset_horiz * geo_.det_pixel_size_horiz;
				auto offset_vert = geo_.det_offset_vert * geo_.det_pixel_size_vert;

				auto sin = 0.f;
				auto cos = 0.f;

				if(!angle_tabs_created_)
				{
					auto angle = img.index() * geo_.rot_angle;
					auto angle_rad = static_cast<float>(angle * M_PI / 180.f);
					sin = std::sin(angle_rad);
					cos = std::cos(angle_rad);
				}
				else
				{
					sin = sin_tab_.at(img.index());
					cos = cos_tab_.at(img.index());
				}

				ddrf::cuda::launch(v.width(), v.height(), v.depth(),
									backproject,
									v.data(), v.width(), v.height(), v.depth(), v.pitch(), vol_offset, vol_geo_.dim_z,
									vol_geo_.voxel_size_x, vol_geo_.voxel_size_y, vol_geo_.voxel_size_z,
									static_cast<const float*>(img.data()), img.width(), img.height(), img.pitch(),
									proj_offset, static_cast<std::size_t>(geo_.det_pixels_column),
									geo_.det_pixel_size_horiz, geo_.det_pixel_size_vert,
									offset_horiz, offset_vert, sin, cos, std::abs(geo_.dist_src), dist_sd_);
			}

		}

		auto Feldkamp::set_input_num(std::uint32_t num) noexcept -> void
		{
			input_num_ = num;
			input_num_set_ = true;
		}

		auto Feldkamp::create_volume(int device) -> void
		{
			CHECK(cudaSetDevice(device));

			auto& scheduler = FeldkampScheduler::instance(geo_, cuda::volume_type::single_float);
			auto vol_dev_size = vol_geo_.dim_z / static_cast<std::size_t>(devices_);
			auto subvol_dev_size = vol_dev_size / scheduler.get_volume_num(device);

			BOOST_LOG_TRIVIAL(debug) << "cuda::Feldkamp: Creating " << vol_geo_.dim_x << "x" << vol_geo_.dim_y << "x" << subvol_dev_size << " volume on device #" << device;
			auto ptr = ddrf::cuda::make_device_ptr<float>(vol_geo_.dim_x, vol_geo_.dim_y, subvol_dev_size);
			ddrf::cuda::memset(ptr, 0);
			volume_map_.emplace(std::make_pair(device, volume_type{ptr.width(), ptr.height(), ptr.depth(), std::move(ptr)}));
		}

		auto Feldkamp::download_and_reset_volume(int device, std::uint32_t vol_num) -> void
		{
			BOOST_LOG_TRIVIAL(info) << "cuda::Feldkamp: Downloading subvolume #" << vol_num << " from device #" << device;
			CHECK(cudaSetDevice(device));

			auto& scheduler = FeldkampScheduler::instance(geo_, cuda::volume_type::single_float);
			auto& v = volume_map_.at(device);
			auto offset = scheduler.get_volume_offset(device, vol_num);

			auto output_start_ptr = output_.data() + offset * output_.width() * output_.height();

			auto parms = cudaMemcpy3DParms{0};
			auto uchar_width = output_.width() * sizeof(float) / sizeof(unsigned char);
			auto height = output_.height();
			parms.srcPtr = make_cudaPitchedPtr(reinterpret_cast<unsigned char*>(v.data()), v.pitch(), uchar_width, height);
			parms.dstPtr = make_cudaPitchedPtr(reinterpret_cast<unsigned char*>(output_start_ptr), output_.pitch(), uchar_width, height);
			parms.extent = make_cudaExtent(uchar_width, height, v.depth());
			parms.kind = cudaMemcpyDeviceToHost;
			CHECK(cudaMemcpy3D(&parms));

			ddrf::cuda::memset(v.container(), 0);
		}
	}
}
