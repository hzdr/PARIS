/*
 * This file is part of the ddafa reconstruction program.
 *
 * Copyright (C) 2016 Helmholtz-Zentrum Dresden-Rossendorf
 *
 * ddafa is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ddafa is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with ddafa. If not, see <http://www.gnu.org/licenses/>.
 *
 * Date: 07 November 2016
 * Authors: Jan Stephan <j.stephan@hzdr.de>
 */

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

#include <boost/program_options.hpp>

#include "geometry.h"
#include "program_options.h"
#include "region_of_interest.h"

namespace ddafa
{
    auto make_program_options(int argc, char** argv) -> program_options
    {
        auto po = program_options{};

        auto geometry_path = std::string{""};

        try
        {
            // General options
            boost::program_options::options_description general{"General options"};
            general.add_options()
                    ("help", "produce a help message")
                    ("geometry-format", "Display geometry file format");

            // Geometry options
            boost::program_options::options_description geo_opts{"Geometry options"};
            geo_opts.add_options()
                    ("geometry", boost::program_options::value<std::string>(&geometry_path)->required(), "Path to geometry file")
                    ("roi", "Region of interest switch (optional)");

            // Region of interest options
            boost::program_options::options_description roi_opts{"Region of Interest options"};
            roi_opts.add_options()
                    ("roi-x1", boost::program_options::value<std::uint32_t>(&po.roi.x1), "leftmost coordinate")
                    ("roi-x2", boost::program_options::value<std::uint32_t>(&po.roi.x2), "rightmost coordinate")
                    ("roi-y1", boost::program_options::value<std::uint32_t>(&po.roi.y1), "uppermost coordinate")
                    ("roi-y2", boost::program_options::value<std::uint32_t>(&po.roi.y2), "lowest coordinate")
                    ("roi-z1", boost::program_options::value<std::uint32_t>(&po.roi.z1), "uppermost slice")
                    ("roi-z2", boost::program_options::value<std::uint32_t>(&po.roi.z2), "lowest slice");

            // I/O options
            boost::program_options::options_description io{"Input/output options"};
            io.add_options()
                    ("input", boost::program_options::value<std::string>(&po.input_path), "Path to projections (optional)")
                    ("output", boost::program_options::value<std::string>(&po.output_path), "Output directory for the reconstructed volume (optional)")
                    ("name", boost::program_options::value<std::string>(&po.prefix)->default_value("vol"), "Name of the reconstructed volume (optional)");

            // Reconstruction options
            boost::program_options::options_description recon{"Reconstruction options"};
            recon.add_options()
                    ("angles", boost::program_options::value<std::string>(&po.angle_path), "Path to projection angles (optional)")
                    ("quality", boost::program_options::value<std::uint16_t>(&po.quality)->default_value(1), "Quality setting (optional)");

            // Geometry file
            boost::program_options::options_description geom{"Geometry file"};
            geom.add_options()
                    ("n_row", boost::program_options::value<std::uint32_t>(&po.det_geo.n_row)->required(), "[integer] number of pixels per detector row (= projection width)")
                    ("n_col", boost::program_options::value<std::uint32_t>(&po.det_geo.n_col)->required(), "[integer] number of pixels per detector column (= projection height)")
                    ("l_px_row", boost::program_options::value<float>(&po.det_geo.l_px_row)->required(), "[float] horizontal pixel size (= distance between pixel centers) in mm")
                    ("l_px_col", boost::program_options::value<float>(&po.det_geo.l_px_col)->required(), "[float] vertical pixel size (= distance between pixel centers) in mm")
                    ("delta_s", boost::program_options::value<float>(&po.det_geo.delta_s)->required(), "[float] horizontal detector offset in pixels")
                    ("delta_t", boost::program_options::value<float>(&po.det_geo.delta_t)->required(), "[float] vertical detector offset in pixels")
                    ("d_so", boost::program_options::value<float>(&po.det_geo.d_so)->required(), "[float] distance between object (= center of rotation) and source in mm")
                    ("d_od", boost::program_options::value<float>(&po.det_geo.d_od)->required(), "[float] distance between object (= center of rotation) and detector in mm")
                    ("delta_phi", boost::program_options::value<float>(&po.det_geo.delta_phi)->required(), "[float] angle step between two successive projections in °");

            // combine
            boost::program_options::options_description params;
            params.add(general).add(geo_opts).add(io).add(recon).add(roi_opts);

            boost::program_options::variables_map param_map, geom_map;
            boost::program_options::store(boost::program_options::parse_command_line(argc, argv, params), param_map);

            if(param_map.count("help"))
            {
                std::cout << params << std::endl;
                std::exit(EXIT_SUCCESS);
            }
            else if(param_map.count("geometry-format"))
            {
                std::cout << geom << std::endl;
                std::exit(EXIT_SUCCESS);
            }

            auto print_missing = [](const char* str)
            {
                std::cerr << "the option '--" << str << "' is required but missing" << std::endl;
                std::exit(EXIT_FAILURE);
            };

            if(param_map.count("input") || param_map.count("output"))
            {
                po.enable_io = true;
                if(param_map.count("input") == 0) print_missing("input");
                if(param_map.count("output") == 0) print_missing("output");
            }

            if(param_map.count("roi"))
            {
                po.enable_roi = true;
                if(param_map.count("roi-x1") == 0) print_missing("roi-x1");
                if(param_map.count("roi-x2") == 0) print_missing("roi-x2");
                if(param_map.count("roi-y1") == 0) print_missing("roi-y1");
                if(param_map.count("roi-y2") == 0) print_missing("roi-y2");
                if(param_map.count("roi-z1") == 0) print_missing("roi-z1");
                if(param_map.count("roi-z2") == 0) print_missing("roi-z2");
            }

            if(param_map.count("angles"))
                po.enable_angles = true;

            boost::program_options::notify(param_map);

            auto&& file = std::ifstream{geometry_path.c_str()};
            if(file)
                boost::program_options::store(boost::program_options::parse_config_file(file, geom), geom_map);
            boost::program_options::notify(geom_map);
        }
        catch(const boost::program_options::error& err)
        {
            std::cerr << err.what() << std::endl;
            std::exit(EXIT_FAILURE);
        }

        return po;
    }
}


