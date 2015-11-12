#include <iostream>

#include "image/Image.h"
#include "image/ImageHandler.h"

#include "master_worker/Master.h"
#include "master_worker/Task.h"
#include "master_worker/Worker.h"

#include "common/Queue.h"

#include "image/implementations/TIFF.h"

#include "cuda/CUDAMaster.h"

int main()
{
	ddafa::image::ImageHandler<ddafa::impl::TIFF> tiff_handler;

	ddafa::master_worker::Master<ddafa::impl::CUDAMaster, int> cuda_master(0);

	std::cout << "Hello, HZDR!" << std::endl;
}
