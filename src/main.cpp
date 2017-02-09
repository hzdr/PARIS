/*
 * This file is part of the PARIS reconstruction program.
 *
 * Copyright (C) 2016 Helmholtz-Zentrum Dresden-Rossendorf
 *
 * PARIS is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PARIS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PARIS. If not, see <http://www.gnu.org/licenses/>.
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

#include <glados/pipeline/task_queue.h>

#include "backend.h"
#include "backprojection.h"
#include "exception.h"
#include "filtering.h"
#include "geometry.h"
#include "loader.h"
#include "make_volume.h"
#include "program_options.h"
#include "sink.h"
#include "source.h"
#include "subvolume_information.h"
#include "task.h"
#include "version.h"
#include "weighting.h"

namespace
{
    auto init_log() -> void
    {
    #ifdef DEBUG
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

    auto reconstruct(glados::pipeline::task_queue<paris::task>* queue,
                     std::size_t task_num,
                     paris::backend::device_handle& device,
                     paris::sink& sink) -> void
    {
        if(queue == nullptr)
            return;

        paris::backend::set_device(device);

        while(!queue->empty())
        {
            auto t = queue->pop();
            auto last = (task_num - t.id) > 1 ? false : true;
            auto source = paris::source(t.input_path, t.enable_angles, t.angle_path, t.quality);

            auto v = paris::make_volume(t.subvol_geo, last);
            auto offset = t.id * t.subvol_geo.dim_z;

            while(!source.drained())
            {
                auto p = source.load_next();
                auto d_p = paris::load(p);
                paris::weight(d_p, t.det_geo);
                paris::filter(d_p, t.det_geo);
                paris::backproject(d_p, v, offset, t.det_geo, t.vol_geo, t.enable_angles, t.enable_roi, t.roi); 
            }
            sink.save(v);
        }

        paris::backend::shutdown();
    }
}

auto main(int argc, char** argv) -> int
{
    std::cout << "PARIS - version " << paris::version << std::endl;
    std::signal(SIGSEGV, signal_handler);
    std::signal(SIGABRT, signal_handler);

    init_log();
    auto po = paris::make_program_options(argc, argv);

    try
    {
        auto vol_geo = paris::calculate_volume_geometry(po.det_geo);

        auto roi_geo = vol_geo;
        if(po.enable_roi)
            roi_geo = paris::apply_roi(vol_geo,
                                       po.roi.x1, po.roi.x2,
                                       po.roi.y1, po.roi.y2,
                                       po.roi.z1, po.roi.z2);

        if(po.enable_io)
        {
            auto start = std::chrono::high_resolution_clock::now();

            // split the volume into subvolumes
            auto subvol_info = paris::backend::make_subvolume_information(roi_geo, po.det_geo);

            // generate tasks
            auto tasks = paris::make_tasks(po, vol_geo, subvol_info);
            auto&& task_queue = glados::pipeline::task_queue<paris::task>(tasks);
            auto task_num = tasks.size();

            // get devices
            auto devices = paris::backend::get_devices();

            // reconstruction futures
            auto futures = std::vector<std::future<void>>{};

            auto task_string = tasks.size() == 1 ? "task" : "tasks";
            auto device_string = devices.size() == 1 ? "device" : "devices";
            BOOST_LOG_TRIVIAL(info) << "Created " << tasks.size() << " " << task_string << " for " << devices.size() << ' ' << device_string;

            // create sink
            auto sink = paris::sink{po.output_path, po.prefix, roi_geo};

            if(devices.size() > 1)
            {
                // launch a reconstruction thread for each available device
                for(auto&& d : devices)
                    futures.emplace_back(std::async(std::launch::async, reconstruct, &task_queue, task_num,
                                                                        std::ref(d), std::ref(sink)));

                // wait for the end of execution
                for(auto&& f : futures)
                    f.get();
            }
            else
                reconstruct(&task_queue, task_num, devices[0], sink);

            auto stop = std::chrono::high_resolution_clock::now();

            auto duration = stop - start;
            auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
            auto minutes = std::chrono::duration_cast<std::chrono::minutes>(duration);

            BOOST_LOG_TRIVIAL(info) << "Program terminated. Time elapsed: "
                    << minutes.count() << ":" << std::setfill('0') << std::setw(2) << seconds.count() % 60 << " minutes";
        }
    }
    catch(const paris::stage_construction_error& sce)
    {
        BOOST_LOG_TRIVIAL(fatal) << "main(): Pipeline construction failed: " << sce.what();
        BOOST_LOG_TRIVIAL(fatal) << "Aborting.";
        std::exit(EXIT_FAILURE);
    }
    catch(const paris::stage_runtime_error& sre)
    {
        BOOST_LOG_TRIVIAL(fatal) << "main(): Pipeline execution failed: " << sre.what();
        BOOST_LOG_TRIVIAL(fatal) << "Aborting.";
        std::exit(EXIT_FAILURE);
    }

    return 0;
}
