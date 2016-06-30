#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

#include <boost/log/trivial.hpp>

#include <ddrf/Image.h>
#include <ddrf/cuda/Check.h>
#include <ddrf/cuda/Coordinates.h>
#include <ddrf/cuda/Launch.h>

#include "../common/Geometry.h"

#include "FeldkampScheduler.h"
#include "Weighting.h"

namespace ddafa
{
    namespace cuda
    {
        __global__ void weight(float* output, const float* input,
                                std::size_t width, std::size_t height, std::size_t pitch, std::size_t offset,
                                float h_min, float v_min, float d_dist,
                                float pixel_size_horiz, float pixel_size_vert)
        {
            auto j = ddrf::cuda::getX(); // column index
            auto i = ddrf::cuda::getY(); // row index

            if((j < width) && (i < height))
            {
                auto input_row = reinterpret_cast<const float*>(reinterpret_cast<const char*>(input) + i * pitch);
                auto output_row = reinterpret_cast<float*>(reinterpret_cast<char*>(output) + i * pitch);

                // remember the subprojection offset
                auto i_off = i + offset;

                // detector coordinates
                auto h_j = (pixel_size_horiz / 2) + j * pixel_size_horiz + h_min;
                auto v_i = (pixel_size_vert / 2) + i_off * pixel_size_vert + v_min;

                // calculate weight
                auto w_ij = d_dist * rsqrtf(powf(d_dist, 2) + powf(h_j, 2) + powf(v_i, 2));

                // apply
                output_row[j] = input_row[j] * w_ij;
            }
        }

        Weighting::Weighting(const common::Geometry& geo)
        {
            auto& scheduler = FeldkampScheduler::instance(geo, volume_type::single_float);
            geo_ = scheduler.get_updated_detector_geometry();
            old_geo_ = geo;
            h_min_ = -(geo_.det_offset_horiz * geo_.det_pixel_size_horiz) - ((static_cast<float>(geo_.det_pixels_row) * geo_.det_pixel_size_horiz) / 2);
            v_min_ = -(geo_.det_offset_vert * geo_.det_pixel_size_vert) - ((static_cast<float>(geo_.det_pixels_column) * geo_.det_pixel_size_vert) / 2);
            d_dist_ = geo_.dist_det + geo_.dist_src;

            CHECK(cudaGetDeviceCount(&devices_));
            for(auto i = 0; i < devices_; ++i)
                processor_threads_[i] = std::thread{&Weighting::processor, this, i};
        }

        auto Weighting::process(input_type&& img) -> void
        {
            if(img.valid())
                map_imgs_[img.device()].push(std::move(img));
            else
            {
                BOOST_LOG_TRIVIAL(debug) << "cuda::Weighting: Received poisonous pill, finishing...";
                for(auto i = 0; i < devices_; ++i)
                    map_imgs_[i].push(input_type());

                for(auto i = 0; i < devices_; ++i)
                    processor_threads_[i].join();

                results_.push(output_type());
                BOOST_LOG_TRIVIAL(info) << "cuda::Weighting: Done.";
            }
        }

        auto Weighting::wait() -> output_type
        {
            return results_.take();
        }

        auto Weighting::processor(int device) -> void
        {
            CHECK(cudaSetDevice(device));

            auto proj_count = 0u;
            auto vol_count = 0u;
            auto& scheduler = FeldkampScheduler::instance(old_geo_, volume_type::single_float);

            while(true)
            {
                auto img = map_imgs_[device].take();
                if(!img.valid())
                    break;

                ++proj_count;
                if(input_num_set_ && (proj_count > input_num_))
                {
                    proj_count = 1u;
                    ++vol_count;
                }

                BOOST_LOG_TRIVIAL(debug) << "cuda::Weighting: processing image #" << img.index() << " on device #" << device;

                auto offset = scheduler.get_subproj_offset(device, vol_count);

                ddrf::cuda::launch(img.width(), img.height(),
                        weight,
                        img.data(), static_cast<const float*>(img.data()), img.width(), img.height(), img.pitch(), offset,
                        h_min_, v_min_, d_dist_, geo_.det_pixel_size_horiz, geo_.det_pixel_size_vert);

                CHECK(cudaStreamSynchronize(0));
                results_.push(std::move(img));
            }
        }

        auto Weighting::set_input_num(std::uint32_t num) -> void
        {
            input_num_ = num;
            input_num_set_ = true;
        }
    }
}
