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
 * Date: 18 August 2016
 * Authors: Jan Stephan <j.stephan@hzdr.de>
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <utility>

#include <boost/log/trivial.hpp>

#include "backend.h"
#include "exception.h"
#include "geometry.h"
#include "projection.h"
#include "reconstruction_constants.h"
#include "reconstruction_stage.h"
#include "region_of_interest.h"
#include "volume.h"

namespace ddafa
{
    namespace
    {
        auto download(const backend::device_ptr_3D<float>& in, backend::host_ptr_3D<float>& out,
                        std::uint32_t x, std::uint32_t y, std::uint32_t z) -> void
        {
            backend::copy(backend::sync, out, in, x, y, z);
        }
    }

    reconstruction_stage::reconstruction_stage(const backend::device_handle& device) noexcept
    : device_{device}
    {
    }

    auto reconstruction_stage::assign_task(task t) noexcept -> void
    {
        det_geo_ = t.det_geo;
        vol_geo_ = t.vol_geo;
        subvol_geo_ = t.subvol_geo;
        enable_angles_ = t.enable_angles;

        enable_roi_ = t.enable_roi;
        roi_ = t.roi;

        task_id_ = t.id;
    }

    auto reconstruction_stage::run() -> void
    {
        auto sre = stage_runtime_error{"reconstruction_stage::run() failed"};

        try
        {
            backend::set_device(device_);

            auto dim_z = std::uint32_t{};
            // if this is the lowest subvolume we need to consider the remaining slices
            if(task_id_ == task_num_ - 1)
                dim_z = subvol_geo_.dim_z + subvol_geo_.remainder;
            else
                dim_z = subvol_geo_.dim_z;

            // create host volume
            auto vol_h_ptr = backend::make_host_ptr<float>(subvol_geo_.dim_x, subvol_geo_.dim_y, dim_z);
            backend::fill(backend::sync, vol_h_ptr, 0, subvol_geo_.dim_x, subvol_geo_.dim_y, dim_z);

            // create device volume
            auto vol_d_ptr = backend::make_device_ptr<float>(subvol_geo_.dim_x, subvol_geo_.dim_y, dim_z);
            backend::fill(backend::sync, vol_d_ptr, 0, subvol_geo_.dim_x, subvol_geo_.dim_y, dim_z);

            // calculate offset for the current subvolume
            auto offset = task_id_ * subvol_geo_.dim_z;

            // utility variables
            auto delta_s = det_geo_.delta_s * det_geo_.l_px_row;
            auto delta_t = det_geo_.delta_t * det_geo_.l_px_col;

            // initialize dev_consts__
            auto host_consts = reconstruction_constants {
                subvol_geo_.dim_x,
                vol_geo_.dim_x,
                subvol_geo_.dim_y,
                vol_geo_.dim_y,
                subvol_geo_.dim_z,
                vol_geo_.dim_z,
                offset,
                vol_geo_.l_vx_x,
                vol_geo_.l_vx_y,
                vol_geo_.l_vx_z,
                det_geo_.n_row,
                det_geo_.n_col,
                det_geo_.l_px_row,
                det_geo_.l_px_col,
                delta_s,
                delta_t,
                det_geo_.d_so,
                std::abs(det_geo_.d_so) + std::abs(det_geo_.d_od)
            };

            auto err = backend::set_reconstruction_constants(host_consts);
            if(err != backend::success)
            {
                backend::print_error("Could not initialize device constants: ", err);
                throw stage_runtime_error{"reconstruction_stage::run() failed"};
            }
            
            // initialize dev_roi__
            if(enable_roi_)
            {
                err = backend::set_roi(roi_);
                if(err != backend::success)
                {
                    backend::print_error("Could not initialize region of interest on device: ", err);
                    throw stage_runtime_error{"reconstruction_stage::process() failed"};
                }
            }

            while(true)
            {
                auto p = input_();
                if(!p.valid)
                    break;

                if(p.idx % 10 == 0)
                    BOOST_LOG_TRIVIAL(info) << "Reconstruction processing projection #" << p.idx << " on device #" << device_;

                // get angular position of the current projection
                auto phi = 0.f;
                if(enable_angles_)
                    phi = p.phi;
                else
                    phi = static_cast<float>(p.idx) * det_geo_.delta_phi;

                // transform to radians
                phi *= static_cast<float>(M_PI) / 180.f;

                auto sin = std::sin(phi);
                auto cos = std::cos(phi);

                backend::backproject(p.async_handle, vol_d_ptr, subvol_geo_.dim_x, subvol_geo_.dim_y, subvol_geo_.dim_z,
                                     p, sin, cos, enable_roi_);


                backend::synchronize(p.async_handle);
            }

            // copy results to host
            download(vol_d_ptr, vol_h_ptr, subvol_geo_.dim_x, subvol_geo_.dim_y, subvol_geo_.dim_z);

            // create and move output volume -- done
            output_(output_type{std::move(vol_h_ptr), subvol_geo_.dim_x, subvol_geo_.dim_y, dim_z, offset, true});
            BOOST_LOG_TRIVIAL(info) << "Completed task #" << task_id_ << " on device #" << device_;
        }
        catch(const backend::bad_alloc& ba)
        {
            BOOST_LOG_TRIVIAL(fatal) << "reconstruction_stage::run() encountered a bad_alloc: " << ba.what();
            throw sre;
        }
        catch(const backend::invalid_argument& ia)
        {
            BOOST_LOG_TRIVIAL(fatal) << "reconstruction_stage::run() passed an invalid argument to the " << backend::name << " runtime: " << ia.what();
            throw sre;
        }
        catch(const backend::runtime_error& re)
        {
            BOOST_LOG_TRIVIAL(fatal) << "reconstruction-stage::run() encountered a " << backend::name << " runtime error: " << re.what();
            throw sre;
        }
    }

    auto reconstruction_stage::set_input_function(std::function<input_type(void)> input) noexcept -> void
    {
        input_ = input;
    }

    auto reconstruction_stage::set_output_function(std::function<void(output_type)> output) noexcept -> void
    {
        output_ = output;
    }
}
