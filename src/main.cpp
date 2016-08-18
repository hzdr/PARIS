#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include <execinfo.h>

#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>
#include <boost/program_options.hpp>

#include <ddrf/pipeline/pipeline.h>

#include "exception.h"
#include "filter_stage.h"
#include "geometry.h"
#include "preloader_stage.h"
#include "source_stage.h"
#include "version.h"
#include "weighting_stage.h"

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

auto main(int argc, char** argv) -> int
{
    std::cout << "ddafa - build " << ddafa::git_build_sha << " from " << ddafa::git_build_time << std::endl;
    std::signal(SIGSEGV, signal_handler);
    std::signal(SIGABRT, signal_handler);

    init_log();

    auto projection_path = std::string{""};
    auto angle_path = std::string{""};
    auto geometry_path = std::string{""};
    auto output_path = std::string{""};
    auto prefix = std::string{""};
    auto geo = ddafa::geometry{};

    try
    {
        // parse parameters
        boost::program_options::options_description param{"Parameters"};
        param.add_options()
                ("help, h", "Help screen")
                ("geometry-format, f", "Display geometry file format")
                ("input, i", boost::program_options::value<std::string>(&projection_path)->required(), "Path to projections")
                ("angles, a", boost::program_options::value<std::string>(&angle_path)->default_value(""), "Path to projection angles (optional)")
                ("geometry, g", boost::program_options::value<std::string>(&geometry_path)->required(), "Path to geometry file")
                ("output, o", boost::program_options::value<std::string>(&output_path)->required(), "Output path for the reconstructed volume")
                ("name, n", boost::program_options::value<std::string>(&prefix)->default_value("vol"), "Name of the reconstructed volume (optional)");

        // parse geometry file
        boost::program_options::options_description geom{"Geometry file"};
        geom.add_options()
                ("det_pixels_row", boost::program_options::value<std::uint32_t>(&geo.det_pixels_row)->required(), "[integer] number of pixels per detector row (= projection width)")
                ("det_pixels_column", boost::program_options::value<std::uint32_t>(&geo.det_pixels_column)->required(), "[integer] number of pixels per detector column (= projection height)")
                ("det_pixel_size_horiz", boost::program_options::value<float>(&geo.det_pixel_size_horiz)->required(), "[float] horizontal pixel size (= distance between pixel centers) in mm")
                ("det_pixel_size_vert", boost::program_options::value<float>(&geo.det_pixel_size_vert)->required(), "[float] vertical pixel size (= distance between pixel centers) in mm")
                ("det_offset_horiz", boost::program_options::value<float>(&geo.det_offset_horiz)->required(), "[float] horizontal detector offset in pixels")
                ("det_offset_vert", boost::program_options::value<float>(&geo.det_offset_vert)->required(), "[float] vertical detector offset in pixels")
                ("dist_src", boost::program_options::value<float>(&geo.dist_src)->required(), "[float] distance between object (= center of rotation) and source in mm")
                ("dist_det", boost::program_options::value<float>(&geo.dist_det)->required(), "[float] distance between object (= center of rotation) and detector in mm")
                ("rot_angle", boost::program_options::value<float>(&geo.rot_angle)->required(), "[float] angle step between two successive projections in °");

        boost::program_options::variables_map param_map, geom_map;
        boost::program_options::store(boost::program_options::parse_command_line(argc, argv, param), param_map);

        if(param_map.count("help"))
        {
            std::cout << param << std::endl;
            return 0;
        }
        else if(param_map.count("geometry-format"))
        {
            std::cout << geom << std::endl;
            return 0;
        }
        boost::program_options::notify(param_map);

        auto&& file = std::ifstream{geometry_path.c_str()};
        if(file)
            boost::program_options::store(boost::program_options::parse_config_file(file, geom), geom_map);
        boost::program_options::notify(geom_map);
    }
    catch(const boost::program_options::error& err)
    {
        std::cerr << err.what() << std::endl;
    }

    try
    {
        // set up pipeline
        auto start = std::chrono::high_resolution_clock::now();

        auto pipeline = ddrf::pipeline::pipeline{};

        auto source = pipeline.create<ddafa::source_stage>(projection_path);
        auto preloader = pipeline.create<ddafa::preloader_stage>();
        auto weighting = pipeline.create<ddafa::weighting_stage>(geo.n_row, geo.n_col, geo.l_px_row, geo.l_px_col, geo.delta_s, geo.delta_t, geo.d_od, geo.d_so);
        auto filter = pipeline.create<ddafa::filter_stage>(geo.n_row, geo.n_col, geo.l_px_row);

        /*
        auto reconstruction = pipeline.create<reconstruction_stage>(geo, angle_path);
        auto sink = pipeline.create<sink_stage>(output_path, prefix);

        pipeline.connect(source, preloader);
        pipeline.connect(preloader, weighting);
        pipeline.connect(weighting, filter);
        pipeline.connect(filter, reconstruction);
        pipeline.connect(reconstruction, sink);

        pipeline.run(source, preloader, weighting, filter, reconstruction, sink);
        weighting->set_input_num(source->num());
        reconstruction->set_input_num(source->num());*/

        pipeline.wait();

        auto stop = std::chrono::high_resolution_clock::now();

        auto duration = stop - start;
        auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
        auto minutes = std::chrono::duration_cast<std::chrono::minutes>(duration);

        BOOST_LOG_TRIVIAL(info) << "Reconstruction finished. Time elapsed: " << minutes.count() << ":" << seconds.count() % 60 << " minutes";
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
