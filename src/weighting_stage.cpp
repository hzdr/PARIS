/*
 * This file is part of the PARIS reconstruction program.
 *
 * Copyright (C) 2016 Helmholtz-Zentrum Dresden-Rossendorf
 *
 * PARIS is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PARIS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PARIS. If not, see <http://www.gnu.org/licenses/>.
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
#include "weighting_stage.h"

namespace paris
{
    weighting_stage::weighting_stage(const backend::device_handle& device) noexcept
    : device_{device}
    {}

    auto weighting_stage::assign_task(task t) noexcept -> void
    {
        det_geo_ = t.det_geo;

        auto n_row_f = static_cast<float>(det_geo_.n_row);
        auto n_col_f = static_cast<float>(det_geo_.n_col);
        h_min_ = det_geo_.delta_s * det_geo_.l_px_row - n_row_f * det_geo_.l_px_row / 2;
        v_min_ = det_geo_.delta_t * det_geo_.l_px_col - n_col_f * det_geo_.l_px_col / 2;
        d_sd_ = std::abs(det_geo_.d_so) + std::abs(det_geo_.d_od);
    }

    auto weighting_stage::run() const -> void
    {
        auto sre = stage_runtime_error{"weighting_stage::run() failed"};

        try
        {
            backend::set_device(device_);
            while(true)
            {
                auto p = input_();
                if(!p.valid)
                    break;

                // weight the projection
                backend::weight(p, h_min_, v_min_, d_sd_, det_geo_.l_px_row, det_geo_.l_px_col);

                // done
                backend::synchronize(p.async_handle);
                output_(std::move(p));
            }

            output_(output_type{});
            BOOST_LOG_TRIVIAL(info) << "Weighted all projections.";
        }
        catch(const backend::bad_alloc& ba)
        {
            BOOST_LOG_TRIVIAL(fatal) << "weighting_stage::run() encountered a bad_alloc: " << ba.what();
            throw sre;
        }
        catch(const backend::invalid_argument& ia)
        {
            BOOST_LOG_TRIVIAL(fatal) << "weighting_stage::run() passed an invalid argument to the " << backend::name << " runtime: " << ia.what();
            throw sre;
        }
        catch(const backend::runtime_error& re)
        {
            BOOST_LOG_TRIVIAL(fatal) << "weighting_stage::run() caused a " << backend::name << " runtime error: " << re.what();
            throw sre;
        }
    }

    auto weighting_stage::set_input_function(std::function<input_type(void)> input) noexcept -> void
    {
        input_ = input;
    }

    auto weighting_stage::set_output_function(std::function<void(output_type)> output) noexcept -> void
    {
        output_ = output;
    }
}
