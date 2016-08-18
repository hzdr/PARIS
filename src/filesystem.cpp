#include <algorithm>
#include <iterator>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <boost/log/trivial.hpp>
#include <boost/filesystem.hpp>

#include "filesystem.h"

namespace ddafa
{
    auto read_directory(const std::string& path) -> std::vector<std::string>
    {
        auto ret = std::vector<std::string>{};
        try
        {
            auto p = boost::filesystem::path{path};
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
            throw std::runtime_error{path + " could not be read"};
        }
        std::sort(std::begin(ret), std::end(ret));
        return ret;
    }

    auto create_directory(const std::string& path) -> bool
    {
        try
        {
            auto p = boost::filesystem::path{path};
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

