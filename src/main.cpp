#include <iostream>
#include <stdexcept>

#include <cuda_runtime.h>

#include "common/Geometry.h"

#include "image/Image.h"
#include "image/ImageHandler.h"
#include "image/handlers/HIS/HIS.h"
#include "image/handlers/TIFF/TIFF.h"

#include "pipeline/Pipeline.h"
#include "pipeline/SinkStage.h"
#include "pipeline/SourceStage.h"
#include "pipeline/Stage.h"

#include "cuda/CUDAFilter.h"
#include "cuda/CUDAToStdImage.h"
#include "cuda/CUDAWeighting.h"

ddafa::common::Geometry createGeometry()
{
	ddafa::common::Geometry geo;

	geo.det_pixels_row = 1016;
	geo.det_pixel_column = 401;
	geo.det_pixel_size_horiz = 0.2f;
	geo.det_pixel_size_vert = 0.2f;
	geo.det_offset_horiz = 4.6f * geo.det_pixel_size_horiz; // 4.6 pixel, nicht mm
	geo.det_offset_vert = 0.0f;

	geo.dist_src = 200;
	geo.dist_det = 100;

	geo.vol_rows = 1024;
	geo.vol_columns = 1024;
	geo.vol_planes = 1024;
	geo.vol_voxel_width = 1;
	geo.vol_voxel_height = 1;
	geo.vol_voxel_depth = 1;

	geo.rot_angle = 0.25f;

	return geo;
}

int main(int argc, char** argv)
{
	using tiff_handler = ddafa::image::ImageHandler<ddafa::impl::TIFF>;
	using his_handler = ddafa::image::ImageHandler<ddafa::impl::HIS>;
	using source_stage = ddafa::pipeline::SourceStage<his_handler>;
	using sink_stage = ddafa::pipeline::SinkStage<tiff_handler>;
	using weighting_stage = ddafa::pipeline::Stage<ddafa::impl::CUDAWeighting, ddafa::common::Geometry>;
	using filter_stage = ddafa::pipeline::Stage<ddafa::impl::CUDAFilter, ddafa::common::Geometry>;
	using converter_stage = ddafa::pipeline::Stage<ddafa::impl::CUDAToStdImage>;

	try
	{
		ddafa::pipeline::Pipeline pipeline;

		auto source = pipeline.create<source_stage>("path");
		auto weighting = pipeline.create<weighting_stage>(createGeometry());
		auto filter = pipeline.create<filter_stage>(createGeometry());
		auto converter = pipeline.create<converter_stage>();
		auto sink = pipeline.create<sink_stage>("path");

		pipeline.connect(source, weighting);
		pipeline.connect(weighting, filter);
		pipeline.connect(filter, converter);
		pipeline.connect(converter, sink);

		pipeline.run(source, weighting, filter, converter, sink);

		pipeline.wait();
	}
	catch(const std::runtime_error& err)
	{
		std::cout << "=========================" << std::endl;
		std::cout << "A runtime error occurred: " << std::endl;
		std::cout << err.what() << std::endl;
		std::cout << "=========================" << std::endl;
	}

	return 0;
}
