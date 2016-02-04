/*
 * Filesystem.cpp
 *
 *  Created on: 04.02.2016
 *      Author: Jan Stephan
 */

#include <algorithm>
#include <iterator>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#define BOOST_ALL_DYN_LINK
#include <boost/log/trivial.hpp>
#include <boost/filesystem.hpp>

#include "Filesystem.h"

namespace ddafa
{
	namespace common
	{
		std::vector<std::string> readDirectory(const std::string& path)
		{
			std::vector<std::string> ret;
			try
			{
				boost::filesystem::path p(path);
				if(boost::filesystem::exists(p))
				{
					if(boost::filesystem::is_regular_file(p))
						throw std::runtime_error(path + " is not a directory.");
					else if(boost::filesystem::is_directory(p))
					{
						for(auto&& it = boost::filesystem::directory_iterator(p);
								it != boost::filesystem::directory_iterator(); ++it)
							ret.push_back(boost::filesystem::canonical(it->path()).string());
					}
					else
						throw std::runtime_error(path + " exists but is neither a regular file nor a directory.");
				}
				else
					throw std::runtime_error(path + " does not exist.");

			}
			catch(const boost::filesystem::filesystem_error& err)
			{
				BOOST_LOG_TRIVIAL(fatal) << path << " could not be read: " << err.what();
			}
			std::sort(std::begin(ret), std::end(ret));
			return ret;
		}

		bool createDirectory(const std::string& path)
		{
			try
			{
				boost::filesystem::path p(path);
				if(boost::filesystem::exists(p))
				{
					if(boost::filesystem::is_directory(p))
						return true;
					else
						throw std::runtime_error(path + " exists but is not a directory.");
				}
				else
					return boost::filesystem::create_directories(p);
			}
			catch(const boost::filesystem::filesystem_error& err)
			{
				BOOST_LOG_TRIVIAL(fatal) << path << " could not be created: " << err.what();
				return false;
			}
		}
	}
}
