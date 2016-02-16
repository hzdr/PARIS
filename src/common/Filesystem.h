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
		auto readDirectory(const std::string&) -> std::vector<std::string>;
		auto createDirectory(const std::string&) -> bool;

	}
}



#endif /* FILESYSTEM_H_ */
