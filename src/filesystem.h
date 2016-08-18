#ifndef DDAFA_FILESYSTEM_H_
#define DDAFA_FILESYSTEM_H_

#include <string>
#include <vector>

namespace ddafa
{
	auto read_directory(const std::string&) -> std::vector<std::string>;
	auto create_directory(const std::string&) -> bool;
}



#endif /* DDAFA_FILESYSTEM_H_ */
