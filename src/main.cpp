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
#include <stdexcept>
#include <string>

#include <execinfo.h>

#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>
#include <boost/program_options.hpp>

#include <ddrf/pipeline/pipeline.h>

#include "device_to_host_stage.h"
#include "exception.h"
#include "filter_stage.h"
#include "geometry.h"
#include "geometry_calculator.h"
#include "preloader_stage.h"
#include "reconstruction_stage.h"
#include "sink_stage.h"
#include "sink_stage_single.h"
#include "source_stage.h"
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

    auto geometry_path = std::string{""};
    auto det_geo = ddafa::detector_geometry{};

    auto enable_io = false;
    auto has_projection_path = false;
    auto has_output_path = false;
    auto projection_path = std::string{""};
    auto output_path = std::string{""};
    auto prefix = std::string{""};

    auto enable_roi = false;
    auto has_roi_x1 = false;
    auto has_roi_x2 = false;
    auto has_roi_y1 = false;
    auto has_roi_y2 = false;
    auto has_roi_z1 = false;
    auto has_roi_z2 = false;
    auto roi_x1 = std::uint32_t{0};
    auto roi_x2 = std::uint32_t{0};
    auto roi_y1 = std::uint32_t{0};
    auto roi_y2 = std::uint32_t{0};
    auto roi_z1 = std::uint32_t{0};
    auto roi_z2 = std::uint32_t{0};

    auto enable_angles = false;
    auto angle_path = std::string{""};

    try
    {
        // General options
        boost::program_options::options_description general{"General options"};
        general.add_options()
                ("help", "produce a help message")
                ("geometry-format", "Display geometry file format");

        // Geometry options
        boost::program_options::options_description geo_opts{"Geometry options"};
        geo_opts.add_options()
                ("geometry", boost::program_options::value<std::string>(&geometry_path)->required(), "Path to geometry file")
                ("roi", "Region of interest switch (optional)");

        // Region of interest options
        boost::program_options::options_description roi_opts{"Region of Interest options"};
        roi_opts.add_options()
                ("roi-x1", boost::program_options::value<std::uint32_t>(&roi_x1), "leftmost coordinate")
                ("roi-x2", boost::program_options::value<std::uint32_t>(&roi_x2), "rightmost coordinate")
                ("roi-y1", boost::program_options::value<std::uint32_t>(&roi_y1), "uppermost coordinate")
                ("roi-y2", boost::program_options::value<std::uint32_t>(&roi_y2), "lowest coordinate")
                ("roi-z1", boost::program_options::value<std::uint32_t>(&roi_z1), "uppermost slice")
                ("roi-z2", boost::program_options::value<std::uint32_t>(&roi_z2), "lowest slice");

        // I/O options
        boost::program_options::options_description io{"Input/output options"};
        io.add_options()
                ("input", boost::program_options::value<std::string>(&projection_path), "Path to projections (optional)")
                ("output", boost::program_options::value<std::string>(&output_path), "Output directory for the reconstructed volume (optional)")
                ("name", boost::program_options::value<std::string>(&prefix)->default_value("vol"), "Name of the reconstructed volume (optional)");

        // Reconstruction options
        boost::program_options::options_description recon{"Reconstruction options"};
        recon.add_options()
                ("angles", boost::program_options::value<std::string>(&angle_path), "Path to projection angles (optional)");

        // Geometry file
        boost::program_options::options_description geom{"Geometry file"};
        geom.add_options()
                ("n_row", boost::program_options::value<std::uint32_t>(&det_geo.n_row)->required(), "[integer] number of pixels per detector row (= projection width)")
                ("n_col", boost::program_options::value<std::uint32_t>(&det_geo.n_col)->required(), "[integer] number of pixels per detector column (= projection height)")
                ("l_px_row", boost::program_options::value<float>(&det_geo.l_px_row)->required(), "[float] horizontal pixel size (= distance between pixel centers) in mm")
                ("l_px_col", boost::program_options::value<float>(&det_geo.l_px_col)->required(), "[float] vertical pixel size (= distance between pixel centers) in mm")
                ("delta_s", boost::program_options::value<float>(&det_geo.delta_s)->required(), "[float] horizontal detector offset in pixels")
                ("delta_t", boost::program_options::value<float>(&det_geo.delta_t)->required(), "[float] vertical detector offset in pixels")
                ("d_so", boost::program_options::value<float>(&det_geo.d_so)->required(), "[float] distance between object (= center of rotation) and source in mm")
                ("d_od", boost::program_options::value<float>(&det_geo.d_od)->required(), "[float] distance between object (= center of rotation) and detector in mm")
                ("delta_phi", boost::program_options::value<float>(&det_geo.delta_phi)->required(), "[float] angle step between two successive projections in °");

        // combine
        boost::program_options::options_description params;
        params.add(general).add(geo_opts).add(io).add(recon).add(roi_opts);

        boost::program_options::variables_map param_map, geom_map;
        boost::program_options::store(boost::program_options::parse_command_line(argc, argv, params), param_map);

        if(param_map.count("help"))
        {
            std::cout << params << std::endl;
            return 0;
        }
        else if(param_map.count("geometry-format"))
        {
            std::cout << geom << std::endl;
            return 0;
        }

        auto print_missing = [](const char* str)
        {
            std::cerr << "the option '--" << str << "' is required but missing" << std::endl;
            std::exit(EXIT_FAILURE);
        };

        if(param_map.count("input") || param_map.count("output"))
        {
            enable_io = true;
            if(param_map.count("input") == 0) print_missing("input");
            if(param_map.count("output") == 0) print_missing("output");
        }

        if(param_map.count("roi"))
        {
            enable_roi = true;
            if(param_map.count("roi-x1") == 0) print_missing("roi-x1");
            if(param_map.count("roi-x2") == 0) print_missing("roi-x2");
            if(param_map.count("roi-y1") == 0) print_missing("roi-y1");
            if(param_map.count("roi-y2") == 0) print_missing("roi-y2");
            if(param_map.count("roi-z1") == 0) print_missing("roi-z1");
            if(param_map.count("roi-z2") == 0) print_missing("roi-z2");
        }

        if(param_map.count("angles"))
            enable_angles = true;

        boost::program_options::notify(param_map);

        auto&& file = std::ifstream{geometry_path.c_str()};
        if(file)
            boost::program_options::store(boost::program_options::parse_config_file(file, geom), geom_map);
        boost::program_options::notify(geom_map);
    }
    catch(const boost::program_options::error& err)
    {
        std::cerr << err.what() << std::endl;
        std::exit(EXIT_FAILURE);
    }

    try
    {
        // auto geo_calc = ddafa::geometry_calculator{geo, enable_roi, roi_x1, roi_x2, roi_y1, roi_y2, roi_z1, roi_z2};
        auto vol_geo = ddafa::calculate_volume_geometry(det_geo, enable_roi, roi_x1, roi_x2, roi_y1, roi_y2, roi_z1, roi_z2);
        auto input_limit = std::size_t{100};

        if(enable_io)
        {
            // set up pipeline
            auto start = std::chrono::high_resolution_clock::now();

            auto pipeline = ddrf::pipeline::pipeline{};

            // uncomment the following when individual projections from the intermediate stages are needed for debugging
            /*auto source = pipeline.make_stage<ddafa::source_stage>(projection_path);
            auto preloader = pipeline.make_stage<ddafa::preloader_stage>(input_limit, input_limit);
            auto weighting = pipeline.make_stage<ddafa::weighting_stage>(input_limit, geo.n_row, geo.n_col, geo.l_px_row, geo.l_px_col, geo.delta_s, geo.delta_t, geo.d_od, geo.d_so);
            auto filter = pipeline.make_stage<ddafa::filter_stage>(input_limit, geo.n_row, geo.n_col, geo.l_px_row);
            auto device_to_host = pipeline.make_stage<ddafa::device_to_host_stage>(input_limit);
            auto sink = pipeline.make_stage<ddafa::sink_stage_single>(output_path, prefix);

            pipeline.connect(source, preloader);
            pipeline.connect(preloader, weighting);
            pipeline.connect(weighting, filter);
            pipeline.connect(filter, device_to_host);
            pipeline.connect(device_to_host, sink);

            pipeline.run(source, preloader, weighting, filter, device_to_host, sink);*/

            auto source = pipeline.make_stage<ddafa::source_stage>(projection_path);
            auto preloader = pipeline.make_stage<ddafa::preloader_stage>(input_limit, input_limit);
            auto weighting = pipeline.make_stage<ddafa::weighting_stage>(input_limit, det_geo.n_row, det_geo.n_col, det_geo.l_px_row, det_geo.l_px_col, geo.delta_s, geo.delta_t, geo.d_od, geo.d_so);
            auto filter = pipeline.make_stage<ddafa::filter_stage>(input_limit, geo.n_row, geo.n_col, geo.l_px_row);
            auto reconstruction = pipeline.make_stage<ddafa::reconstruction_stage>(input_limit, geo, geo_calc.get_volume_metadata(), geo_calc.get_subvolume_metadata(), enable_angles);
            auto sink = pipeline.make_stage<ddafa::sink_stage>(output_path, prefix);

            pipeline.connect(source, preloader);
            pipeline.connect(preloader, weighting);
            pipeline.connect(weighting, filter);
            pipeline.connect(filter, reconstruction);
            pipeline.connect(reconstruction, sink);

            pipeline.run(source, preloader, weighting, filter, reconstruction, sink);

            pipeline.wait();

            auto stop = std::chrono::high_resolution_clock::now();

            auto duration = stop - start;
            auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
            auto minutes = std::chrono::duration_cast<std::chrono::minutes>(duration);

            BOOST_LOG_TRIVIAL(info) << "Reconstruction finished. Time elapsed: " << minutes.count() << ":" << std::setfill('0') << std::setw(2) << seconds.count() % 60 << " minutes";
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
