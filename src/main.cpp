/*
 * This file is part of the ddafa reconstruction program.
 *
 * Copyright (C) 2016 Helmholtz-Zentrum Dresden-Rossendorf
 *
 * ddafa is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ddafa is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with ddafa. If not, see <http://www.gnu.org/licenses/>.
 *
 * Date: 18 August 2016
 * Authors: Jan Stephan <j.stephan@hzdr.de>
 */

#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <future>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <execinfo.h>

#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>

#include <glados/pipeline/pipeline.h>

#include "backend.h"
#include "exception.h"
#include "filter_stage.h"
#include "geometry.h"
#include "preloader_stage.h"
#include "program_options.h"
#include "reconstruction_stage.h"
#include "sink_stage.h"
#include "source_stage.h"
#include "subvolume_information.h"
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

    auto launch_pipeline(glados::pipeline::task_queue<ddafa::task>* queue, int device,
                         glados::pipeline::stage<ddafa::sink_stage>& sink, std::size_t input_limit,
                         std::size_t parallel_projections) -> void
    {
        if(queue == nullptr)
            return;

        auto pipeline = glados::pipeline::task_pipeline<ddafa::task>{queue};
        auto source = pipeline.make_stage<ddafa::source_stage>();
        auto preloader = pipeline.make_stage<ddafa::preloader_stage>(input_limit, parallel_projections, device);
        auto weighting = pipeline.make_stage<ddafa::weighting_stage>(input_limit, device);
        auto filter = pipeline.make_stage<ddafa::filter_stage>(input_limit, device);
        auto reconstruction = pipeline.make_stage<ddafa::reconstruction_stage>(input_limit, device);

        pipeline.connect(source, preloader, weighting, filter, reconstruction, sink);
        pipeline.run(source, preloader, weighting, filter, reconstruction);
        pipeline.wait();
    }
}

auto main(int argc, char** argv) -> int
{
    std::cout << "ddafa - version " << ddafa::version << std::endl;
    std::signal(SIGSEGV, signal_handler);
    std::signal(SIGABRT, signal_handler);

    init_log();
    auto po = ddafa::make_program_options(argc, argv);

    try
    {
        constexpr auto parallel_projections = std::size_t{5}; // number of projections present in the pipeline at the same time
        constexpr auto input_limit = std::size_t{1}; // input limit per stage

        auto vol_geo = ddafa::calculate_volume_geometry(po.det_geo);

        auto roi_geo = vol_geo;
        if(po.enable_roi)
            roi_geo = ddafa::apply_roi(vol_geo,
                                       po.roi.x1, po.roi.x2,
                                       po.roi.y1, po.roi.y2,
                                       po.roi.z1, po.roi.z2);

        if(po.enable_io)
        {
            auto start = std::chrono::high_resolution_clock::now();

            // split the volume into subvolumes
            auto subvol_info = ddafa::backend::make_subvolume_information(roi_geo, po.det_geo, parallel_projections);

            // generate tasks
            auto tasks = ddafa::make_tasks(po, vol_geo, subvol_info);
            auto&& task_queue = glados::pipeline::task_queue<ddafa::task>(tasks);

            // get devices
            auto devices = ddafa::backend::get_devices();

            // create shared sink
            auto sink = glados::pipeline::stage<ddafa::sink_stage>(po.output_path, po.prefix, roi_geo, tasks.size());

            // pipeline futures
            auto futures = std::vector<std::future<void>>{};

            auto task_string = tasks.size() == 1 ? "task" : "tasks";
            BOOST_LOG_TRIVIAL(info) << "Created " << tasks.size() << " " << task_string << " for " << devices.size() << " devices";

            // launch a pipeline for each available device
            for(auto&& d : devices)
                futures.emplace_back(std::async(std::launch::async, launch_pipeline,
                                                &task_queue, d, std::ref(sink), input_limit, parallel_projections));

            auto sink_future = std::async(std::launch::async, &glados::pipeline::stage<ddafa::sink_stage>::run, &sink);

            // wait for the end of execution
            for(auto&& f : futures)
                f.get();

            sink_future.get();

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
