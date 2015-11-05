#include <iostream>

#include "Image.h"
#include "ImageHandler.h"

#include "ImageHandlerPolicies/TIFFHandler.h"

int main()
{
	ImageHandler<TIFFHandler> tiffHandler;
	std::cout << "Hello, HZDR!" << std::endl;
}
