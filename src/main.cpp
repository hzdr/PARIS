#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

#define BOOST_ALL_DYN_LINK
#include <boost/log/core.hpp>
#include <boost/program_options.hpp>

#include "common/Geometry.h"

#include "image/Image.h"
#include "image/ImageLoader.h"
#include "image/ImageSaver.h"
#include "image/loaders/HIS.h"
#include "image/savers/TIFF.h"

#include "pipeline/Pipeline.h"
#include "pipeline/SinkStage.h"
#include "pipeline/SourceStage.h"
#include "pipeline/Stage.h"

#include "cuda/CUDAFilter.h"
#include "cuda/CUDAHostAllocator.h"
#include "cuda/CUDAHostDeleter.h"
#include "cuda/CUDAToStdImage.h"
#include "cuda/CUDAWeighting.h"

void initLog()
{
#ifndef DDAFA_DEBUG
	boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::info);
#endif
}

int main(int argc, char** argv)
{
	initLog();
	using tiff_saver = ddafa::image::ImageSaver<ddafa::impl::TIFF<float, ddafa::impl::CUDAHostAllocator<float>, ddafa::impl::CUDAHostDeleter>>;
	using his_loader = ddafa::image::ImageLoader<ddafa::impl::HIS<float, ddafa::impl::CUDAHostAllocator<float>, ddafa::impl::CUDAHostDeleter>>;
	using source_stage = ddafa::pipeline::SourceStage<his_loader>;
	using sink_stage = ddafa::pipeline::SinkStage<tiff_saver>;
	using weighting_stage = ddafa::pipeline::Stage<ddafa::impl::CUDAWeighting>;
	using filter_stage = ddafa::pipeline::Stage<ddafa::impl::CUDAFilter>;
	using converter_stage = ddafa::pipeline::Stage<ddafa::impl::CUDAToStdImage>;

	try
	{
		std::string projection_path;
		std::string geometry_path;
		std::string output_path;
		std::string prefix;
		ddafa::common::Geometry geo;

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

		std::ifstream file{geometry_path.c_str()};
		if(file)
			boost::program_options::store(boost::program_options::parse_config_file(file, geom), geom_map);
		boost::program_options::notify(geom_map);

		// set up pipeline
		ddafa::pipeline::Pipeline pipeline;

		auto source = pipeline.create<source_stage>(projection_path);
		auto weighting = pipeline.create<weighting_stage>(geo);
		auto filter = pipeline.create<filter_stage>(geo);
		auto converter = pipeline.create<converter_stage>();
		auto sink = pipeline.create<sink_stage>(output_path, prefix);

		pipeline.connect(source, weighting);
		pipeline.connect(weighting, filter);
		pipeline.connect(filter, converter);
		pipeline.connect(converter, sink);

		pipeline.run(source, weighting, filter, converter, sink);
		// pipeline.run(source, weighting, converter, sink);

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
