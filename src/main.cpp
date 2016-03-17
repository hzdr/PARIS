#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

#define BOOST_ALL_DYN_LINK
#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>
#include <boost/program_options.hpp>

#include "common/Geometry.h"

#include <ddrf/Image.h>
#include <ddrf/ImageLoader.h>
#include <ddrf/ImageSaver.h>
#include <ddrf/imageLoaders/HIS/HIS.h>
#include <ddrf/imageSavers/TIFF/TIFF.h>

#include <ddrf/pipeline/Pipeline.h>
#include <ddrf/pipeline/SinkStage.h>
#include <ddrf/pipeline/SourceStage.h>
#include <ddrf/pipeline/Stage.h>

#include <ddrf/cuda/HostMemoryManager.h>

#include "cuda/Feldkamp.h"
#include "cuda/Filter.h"
#include "cuda/Preloader.h"
#include "cuda/ToHostImage.h"
#include "cuda/Weighting.h"

#include "cuda/FeldkampScheduler.h"

void initLog()
{
#ifdef DDAFA_DEBUG
	boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::debug);
#else
	boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::info);
#endif
}

int main(int argc, char** argv)
{
	initLog();
	using tiff_saver = ddrf::ImageSaver<ddrf::savers::TIFF<float, ddrf::cuda::HostMemoryManager<float>>>;
	using his_loader = ddrf::ImageLoader<ddrf::loaders::HIS<float, ddrf::cuda::HostMemoryManager<float>>>;
	using source_stage = ddrf::pipeline::SourceStage<his_loader>;
	using sink_stage = ddrf::pipeline::SinkStage<tiff_saver>;
	using weighting_stage = ddrf::pipeline::Stage<ddafa::cuda::Weighting>;
	using filter_stage = ddrf::pipeline::Stage<ddafa::cuda::Filter>;
	using converter_stage = ddrf::pipeline::Stage<ddafa::cuda::ToHostImage>;
	using preloader_stage = ddrf::pipeline::Stage<ddafa::cuda::Preloader>;
	using reconstruction_stage = ddrf::pipeline::Stage<ddafa::cuda::Feldkamp>;

	try
	{
		auto projection_path = std::string{""};
		auto geometry_path = std::string{""};
		auto output_path = std::string{""};
		auto prefix = std::string{""};
		auto geo = ddafa::common::Geometry{};

		// parse parameters
		boost::program_options::options_description param{"Parameters"};
		param.add_options()
				("help, h", "Help screen")
				("geometry-format, f", "Display geometry file format")
				("input, i", boost::program_options::value<std::string>(&projection_path)->required(), "Path to projections")
				("geometry, g", boost::program_options::value<std::string>(&geometry_path)->required(), "Path to geometry file")
				("output, o", boost::program_options::value<std::string>(&output_path)->required(), "Output path for the reconstructed volume")
				("name, n", boost::program_options::value<std::string>(&prefix)->default_value("vol"), "Name of the reconstructed volume (optional)");

		// parse geometry file
		boost::program_options::options_description geom{"Geometry file"};
		geom.add_options()
				("det_pixels_row", boost::program_options::value<std::uint32_t>(&geo.det_pixels_row)->required(), "Detector pixels per row")
				("det_pixels_column", boost::program_options::value<std::uint32_t>(&geo.det_pixels_column)->required(), "Detector pixels per column")
				("det_pixel_size_horiz", boost::program_options::value<float>(&geo.det_pixel_size_horiz)->required(), "Detector pixel size (= size between pixel centers) in horizontal direction")
				("det_pixel_size_vert", boost::program_options::value<float>(&geo.det_pixel_size_vert)->required(), "Detector pixel size (= size between pixel centers) in vertical direction")
				("det_offset_horiz", boost::program_options::value<float>(&geo.det_offset_horiz)->required(), "Detector offset in horizontal direction")
				("det_offset_vert", boost::program_options::value<float>(&geo.det_offset_vert)->required(), "Detector offset in vertical direction")
				("dist_src", boost::program_options::value<float>(&geo.dist_src)->required(), "Distance between object (= center of rotation) and source")
				("dist_det", boost::program_options::value<float>(&geo.dist_det)->required(), "Distance between detector and source")
				("rot_angle", boost::program_options::value<float>(&geo.rot_angle)->required(), "Angle of rotation");

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

		// set up pipeline
		auto pipeline = ddrf::pipeline::Pipeline{};

		auto source = pipeline.create<source_stage>(projection_path);
		auto preloader = pipeline.create<preloader_stage>(geo);
		auto weighting = pipeline.create<weighting_stage>(geo);
		auto filter = pipeline.create<filter_stage>(geo);
		auto reconstruction = pipeline.create<reconstruction_stage>(geo);
		auto converter = pipeline.create<converter_stage>();
		auto sink = pipeline.create<sink_stage>(output_path, prefix);

		pipeline.connect(source, preloader);
		pipeline.connect(preloader, weighting);
		pipeline.connect(weighting, filter);
		pipeline.connect(filter, converter);
		// pipeline.connect(preloader, converter);
		pipeline.connect(converter, sink);

		pipeline.run(source, preloader, weighting, filter, converter, sink);
		// pipeline.run(source, preloader, converter, sink);
		reconstruction->set_input_num(source->num());

		pipeline.wait();
	}
	catch(const std::runtime_error& err)
	{
		std::cerr << "=========================" << std::endl;
		std::cerr << "A runtime error occurred: " << std::endl;
		std::cerr << err.what() << std::endl;
		std::cerr << "=========================" << std::endl;
	}
	catch(const boost::program_options::error& err)
	{
		std::cerr << err.what() << std::endl;
	}

	return 0;
}
