#include <iostream>

#include "Image.h"
#include "ImageHandler.h"

#include "ImageHandlerPolicies/TIFFHandler.h"

int main()
{
	ImageHandler<TIFFHandler> tiff_handler;

	std::cout << "Hello, HZDR!" << std::endl;
}
