#include <iostream>

#include "common/Geometry.h"

#include "image/Image.h"
#include "image/ImageHandler.h"
#include "image/handlers/TIFF.h"

#include "pipeline/Pipeline.h"
#include "pipeline/SinkStage.h"
#include "pipeline/SourceStage.h"
#include "pipeline/Stage.h"

#include "cuda/CUDAWeighting.h"

ddafa::common::Geometry createGeometry()
{
	ddafa::common::Geometry geo;

	geo.det_pixels_row = 1024;
	geo.det_pixel_column = 1024;
	geo.det_pixel_size_horiz = 1;
	geo.det_pixel_size_vert = 1;
	geo.det_offset_horiz = 0;
	geo.det_offset_vert = 0;

	geo.dist_src = 200;
	geo.dist_det = 100;

	geo.vol_rows = 1024;
	geo.vol_columns = 1024;
	geo.vol_planes = 1024;
	geo.vol_voxel_width = 1;
	geo.vol_voxel_height = 1;
	geo.vol_voxel_depth = 1;

	return geo;
}

int main()
{
	using tiff_handler = ddafa::image::ImageHandler<ddafa::impl::TIFF>;
	using source_stage = ddafa::pipeline::SourceStage<tiff_handler>;
	using sink_stage = ddafa::pipeline::SinkStage<tiff_handler>;
	using weighting_stage = ddafa::pipeline::Stage<ddafa::impl::CUDAWeighting>;

	ddafa::pipeline::Pipeline pipeline;

	auto source = pipeline.create<source_stage>("path");
	//auto weighting = pipeline.create<weighting_stage>(createGeometry());
	auto sink = pipeline.create<sink_stage>("path");

	//ddafa::pipeline::connect(source, weighting);
	//ddafa::pipeline::connect(weighting, sink);
	pipeline.connect(source, sink);

	pipeline.run(source, sink);
	pipeline.wait();

	std::cout << "Hello, HZDR!" << std::endl;
}
