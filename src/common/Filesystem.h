/*
 * Filesystem.h
 *
 *  Created on: 04.02.2016
 *      Author: Jan Stephan
 */

#ifndef FILESYSTEM_H_
#define FILESYSTEM_H_

#include <string>
#include <vector>

namespace ddafa
{
	namespace common
	{
		std::vector<std::string> readDirectory(const std::string&);

	}
}



#endif /* FILESYSTEM_H_ */
