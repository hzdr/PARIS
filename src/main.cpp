#include <iostream>

#include "image/Image.h"
#include "image/ImageHandler.h"

#include "master_worker/Master.h"
#include "master_worker/Task.h"
#include "master_worker/Worker.h"

#include "common/Queue.h"

#include "image/implementations/TIFFHandler.h"

#include "cuda/CUDAMaster.h"

int main()
{
	ImageHandler<TIFFHandler> tiff_handler;

	Master<CUDAMaster, int> cuda_master(0);

	std::cout << "Hello, HZDR!" << std::endl;
}
