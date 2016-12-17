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
#include <numeric>
#include <utility>

#include <boost/log/trivial.hpp>

#include "backend.h"
#include "exception.h"
#include "geometry.h"
#include "filter_stage.h"

namespace paris
{
    filter_stage::filter_stage(const backend::device_handle& device) noexcept
    : device_{device}
    {}

    auto filter_stage::assign_task(task t) noexcept -> void
    {
        filter_size_ = static_cast<std::uint32_t>(2 * std::pow(2, std::ceil(std::log2(t.det_geo.n_row))));
        n_col_ = t.det_geo.n_col;
        tau_ = t.det_geo.l_px_row;
    }

    auto filter_stage::run() -> void
    {
        auto sre = stage_runtime_error{"filter_stage::run() failed"};

        try
        {
            backend::set_device(device_);

            // create filter
            auto k = backend::make_filter(filter_size_, tau_);

            // dimensionality of the FFT - 1D in this case
            constexpr auto rank = 1;

            // size of the FFT for each dimension
            auto n = static_cast<int>(filter_size_);

            // we are executing a batched FFT -> set batch size
            auto batch = static_cast<int>(n_col_);

            // allocate memory for expanded projection (projection width -> filter_size_)
            auto p_exp = backend::make_device_ptr<float>(filter_size_, n_col_);

            // allocate memory for transformed projection (filter_size_ -> size_trans)
            auto size_trans = filter_size_ / 2 + 1;
            auto p_trans = backend::make_device_ptr<backend::fft::complex_type>(size_trans, n_col_);

            // calculate the distance between the first elements of two successive lines
            auto p_exp_dist = backend::calculate_distance(p_exp);
            auto p_trans_dist = backend::calculate_distance(p_trans);

            // set the distance between two successive elements
            constexpr auto p_exp_stride = 1;
            constexpr auto p_trans_stride = 1;

            // set storage dimensions of data in memory
            auto p_exp_nembed = p_exp_dist;
            auto p_trans_nembed = p_trans_dist;

            // create plans for forward and inverse FFT
            auto forward = backend::fft::make_plan<backend::fft::r2c>(rank, &n, batch,
                                                        p_exp.get(), &p_exp_nembed, p_exp_stride, p_exp_dist,
                                                        p_trans.get(), &p_trans_nembed, p_trans_stride, p_trans_dist);

            auto inverse = backend::fft::make_plan<backend::fft::c2r>(rank, &n, batch,
                                                        p_trans.get(), &p_trans_nembed, p_trans_stride, p_trans_dist,
                                                        p_exp.get(), &p_exp_nembed, p_exp_stride, p_exp_dist);

            backend::synchronize(backend::default_async_handle);

            BOOST_LOG_TRIVIAL(debug) << "Filter setup on device #" << device_ << " completed.";

            while(true)
            {
                auto p = input_();
                if(!p.valid)
                    break;

                // expand and transform the projection
                backend::expand(p, p_exp, filter_size_, n_col_);
                backend::transform(p_exp, p_trans, forward, p.async_handle);

                // apply the filter to the transformed projection
                backend::apply_filter(p_trans, k, size_trans, n_col_, p.async_handle);

                // inverse transformation
                backend::transform(p_trans, p_exp, inverse, p.async_handle);

                // shrink to original size and normalize
                backend::shrink(p_exp, p);
                backend::normalize(p, filter_size_);

                // done
                backend::synchronize(p.async_handle);
                output_(std::move(p));
            }

            output_(output_type{});
            BOOST_LOG_TRIVIAL(info) << "All projections have been filtered.";
        }
        catch(const backend::bad_alloc& ba)
        {
            BOOST_LOG_TRIVIAL(fatal) << "filter_stage::run() encountered a bad_alloc: " << ba.what();
            throw sre;
        }
        catch(const backend::invalid_argument& ia)
        {
            BOOST_LOG_TRIVIAL(fatal) << "filter_stage::run() passed an invalid argument to the " << backend::name << " runtime: " << ia.what();
            throw sre;
        }
        catch(const backend::runtime_error& re)
        {
            BOOST_LOG_TRIVIAL(fatal) << "filter_stage::run() caused a " << backend::name << " runtime error: " << re.what();
            throw sre;
        }
        catch(const backend::fft::bad_alloc& ba)
        {
            BOOST_LOG_TRIVIAL(fatal) << "filter_stage::run() encountered a bad_alloc in " << backend::fft::name << ": " << ba.what();
            throw sre;
        }
        catch(const backend::fft::invalid_argument& ia)
        {
            BOOST_LOG_TRIVIAL(fatal) << "filter_stage::run() passed an invalid argument to " << backend::fft::name << ": " << ia.what();
            throw sre;
        }
        catch(const backend::fft::runtime_error& re)
        {
            BOOST_LOG_TRIVIAL(fatal) << "filter_stage::run() encountered a " << backend::fft::name << " runtime error: " << re.what();
            throw sre;
        }
    }

    auto filter_stage::set_input_function(std::function<input_type(void)> input) noexcept -> void
    {
        input_ = input;
    }

    auto filter_stage::set_output_function(std::function<void(output_type)> output) noexcept -> void
    {
        output_ = output;
    }
}
