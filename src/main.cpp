#include <iostream>

#include "Image.h"
#include "ImageHandler.h"

#include "InputSide.h"
#include "OutputSide.h"
#include "Queue.h"

#include "SourceStage.h"
#include "SinkStage.h"
#include "Stage.h"
#include "Pipeline.h"

#include "ImageHandlerPolicies/TIFFHandler.h"

int main()
{
	ImageHandler<TIFFHandler> tiff_handler;
	std::cout << "Hello, HZDR!" << std::endl;
}
