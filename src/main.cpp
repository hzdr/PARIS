/*
 * This file is part of the ddafa reconstruction program.
 *
 * Copyright (C) 2016 Helmholtz-Zentrum Dresden-Rossendorf
 *
 * Licensed under the EUPL, Version 1.1 or - as soon they will be approved by
 * the European Commission - subsequent version of the EUPL (the "Licence");
 * You may not use this work except in compliance with the Licence.
 * You may obtain a copy of the Licence at:
 *
 * http://ec.europa.eu/idabc/eupl
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the Licence is distributed on an "AS IS" basis,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the Licence for the specific language governing permissions and
 * limitations under the Licence.
 *
 * Date: 18 August 2016
 * Authors: Jan Stephan
 */

#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

#include <execinfo.h>

#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>

#include <ddrf/cuda/utility.h>
#include <ddrf/pipeline/pipeline.h>

#include "exception.h"
#include "filter_stage.h"
#include "geometry.h"
#include "preloader_stage.h"
#include "program_options.h"
#include "reconstruction_stage.h"
#include "scheduler.h"
#include "sink_stage.h"
#include "source_stage.h"
#include "task.h"
#include "version.h"
#include "weighting_stage.h"

namespace
{
    auto init_log() -> void
    {
    #ifdef DDAFA_DEBUG
        boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::debug);
    #else
        boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::info);
    #endif
    }

    [[noreturn]] auto signal_handler(int sig) -> void
    {
        void* array[10];
        auto size = backtrace(array, 10);

        BOOST_LOG_TRIVIAL(error) << "Signal " << sig;
        backtrace_symbols_fd(array, size, STDERR_FILENO);
        std::exit(EXIT_FAILURE);
    }
}

auto main(int argc, char** argv) -> int
{
    std::cout << "ddafa - version " << ddafa::version << " from " << ddafa::git_build_time << std::endl;
    std::signal(SIGSEGV, signal_handler);
    std::signal(SIGABRT, signal_handler);

    init_log();
    auto po = ddafa::make_program_options(argc, argv);

    try
    {
        constexpr auto parallel_projections = 5; // number of projections present in the pipeline at the same time
        constexpr auto input_limit = std::size_t{1}; // input limit per stage

        auto vol_geo = ddafa::calculate_volume_geometry(po.det_geo, po.enable_roi,
                                                        po.roi.x1, po.roi.x2,
                                                        po.roi.y1, po.roi.y2,
                                                        po.roi.z1, po.roi.z2);
        auto subvol_info = ddafa::create_subvolume_information(vol_geo, po.det_geo, parallel_projections);

        if(po.enable_io)
        {
            auto start = std::chrono::high_resolution_clock::now();

            // generate tasks
            auto tasks = ddafa::make_tasks(po, vol_geo, subvol_info);
            auto&& task_queue = ddrf::pipeline::task_queue<ddafa::task>(tasks);

            // get number of devices
            auto devices = ddrf::cuda::get_device_count();

            // create shared sink and register devices
            auto sink = ddrf::pipeline::stage<ddafa::sink_stage>(po.output_path, po.prefix);

            // set up and run pipelines
            std::map<int, ddrf::pipeline::task_pipeline<ddafa::task>> pipelines;

            for(auto d = 0; d < devices; ++d)
            {
                pipelines.emplace(d, ddrf::pipeline::task_pipeline<ddafa::task>{&task_queue});

                auto&& pipeline = pipelines.at(d);
                auto source = pipeline.make_stage<ddafa::source_stage>();
                auto preloader = pipeline.make_stage<ddafa::preloader_stage>(input_limit, parallel_projections, d);
                auto weighting = pipeline.make_stage<ddafa::weighting_stage>(input_limit, d);
                auto filter = pipeline.make_stage<ddafa::filter_stage>(input_limit, d);
                auto reconstruction = pipeline.make_stage<ddafa::reconstruction_stage>(input_limit, d);

                pipeline.connect(source, preloader, weighting, filter, reconstruction, sink);
                pipeline.run(source, preloader, weighting, filter, reconstruction, sink);
            }

            // wait for the end of execution
            for(auto d = 0; d < devices; ++d)
                pipelines.at(d).wait();

            auto stop = std::chrono::high_resolution_clock::now();

            auto duration = stop - start;
            auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
            auto minutes = std::chrono::duration_cast<std::chrono::minutes>(duration);

            BOOST_LOG_TRIVIAL(info) << "Reconstruction finished. Time elapsed: "
                    << minutes.count() << ":" << std::setfill('0') << std::setw(2) << seconds.count() % 60 << " minutes";
        }
    }
    catch(const ddafa::stage_construction_error& sce)
    {
        BOOST_LOG_TRIVIAL(fatal) << "main(): Pipeline construction failed: " << sce.what();
        BOOST_LOG_TRIVIAL(fatal) << "Aborting.";
        std::exit(EXIT_FAILURE);
    }
    catch(const ddafa::stage_runtime_error& sre)
    {
        BOOST_LOG_TRIVIAL(fatal) << "main(): Pipeline execution failed: " << sre.what();
        BOOST_LOG_TRIVIAL(fatal) << "Aborting.";
        std::exit(EXIT_FAILURE);
    }

    return 0;
}
