#include <functional>
#include <utility>

#include <boost/log/trivial.hpp>

#include <ddrf/cuda/algorithm.h>
#include <ddrf/cuda/sync_policy.h>

#include "exception.h"
#include "metadata.h"
#include "preloader_stage.h"

namespace ddafa
{
    auto preloader_stage::run() -> void
    {
        auto devices = int{};
        auto err = cudaGetDeviceCount(&devices);
        if(err != cudaSuccess)
        {
            BOOST_LOG_TRIVIAL(fatal) << "preloader_stage::run() could not obtain CUDA devices: " << cudaGetErrorString(err);
            throw stage_runtime_error{"preloader_stage::run() failed to initialize"};
        }

        while(true)
        {
            auto proj = input_();

            if(!proj.second.valid)
                break;

            for(auto i = 0; i < devices; ++i)
            {
                err = cudaSetDevice(i);
                if(err != cudaSuccess)
                {
                    BOOST_LOG_TRIVIAL(fatal) << "preloader_stage::run() could not set CUDA device: " << cudaGetErrorString(err);
                    throw stage_runtime_error{"preloader_stage::run() failed to initialize"};
                }

                auto dev_proj = alloc_.smart_allocate(proj.second.width, proj.second.height);
                try
                {
                    ddrf::cuda::copy(ddrf::cuda::async, dev_proj, proj.first, proj.second.width, proj.second.height);
                }
                catch(const ddrf::cuda::invalid_argument& ia)
                {
                    BOOST_LOG_TRIVIAL(fatal) << "preloader_stage::run() could not copy to CUDA device: " << ia.what();
                    throw stage_runtime_error{"preloader_stage::run() failed to copy projection"};
                }

                auto meta = projection_metadata{proj.second.width, proj.second.height, proj.second.index, proj.second.phi, true, i};
                output_(std::make_pair(std::move(dev_proj), meta));
            }
        }

        // Uploaded all projections to the GPU, notify the next stage that we are done here
        output_(std::make_pair(nullptr, projection_metadata{0, 0, 0, 0.f, false}));
        BOOST_LOG_TRIVIAL(info) << "Uploaded all projections to the device(s)";
    }

    auto preloader_stage::set_input_function(std::function<input_type(void)> input) noexcept -> void
    {
        input_ = input;
    }

    auto preloader_stage::set_output_function(std::function<void(output_type)> output) noexcept -> void
    {
        output_ = output;
    }
}
