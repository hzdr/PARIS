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
 * Date: 05 December 2016
 * Authors: Jan Stephan <j.stephan@hzdr.de>
 */

#include <cmath>
#include <cstdint>
#include <memory>
#include <numeric>

#include <fftw3.h>

#include "backend.h"

namespace paris
{
    namespace openmp
    {
        namespace
        {
            template <class T>
            auto make_ptr(std::uint32_t dim) -> std::unique_ptr<T[], fftw_deleter>
            {
                auto p = reinterpret_cast<T*>(fftwf_malloc(dim * sizeof(T)));
                return std::unique_ptr<T[], fftw_deleter>{p};
            }

            template <class T>
            auto make_ptr(std::uint32_t dim_x, std::uint32_t dim_y) -> std::unique_ptr<T[], fftw_deleter>
            {
                auto p = reinterpret_cast<T*>(fftwf_malloc(dim_x * dim_y * sizeof(T)));
                return std::unique_ptr<T[], fftw_deleter>{p};
            }

            auto make_filter_real(float* r, std::uint32_t size, float tau) -> void
            {
                auto js = std::make_unique<std::int32_t[]>(size);
                auto j = -(static_cast<std::int32_t>(size) - 2) / 2;
                std::iota(js.get(), js.get() + size, j);

                auto pi_f = static_cast<float>(M_PI);

                #pragma omp parallel for
                for(auto x = 0u; x < size; ++x)
                {
                    if(js[x] == 0)
                        r[x] = (1.f / 8.f) * (1.f / std::pow(tau, 2.f));
                    else
                    {
                        if(js[x] % 2 == 0)
                            r[x] = 0.f;
                        else
                            r[x] = -(1.f / (2.f * static_cast<float>(js[x] * js[x]) * (pi_f * pi_f) * (tau * tau)));
                    }
                }
            }   

            auto expand(const float* src, std::uint32_t src_dim_x,
                              float* dst, std::uint32_t dst_dim_x, std::uint32_t dim_y) noexcept -> void
            {
                // copy original projection to expanded projection
                #pragma omp parallel for collapse(2)
                for(auto y = 0u; y < dim_y; ++y)
                {
                    for(auto x = 0u; x < dst_dim_x; ++x)
                    {
                        if(x < src_dim_x)
                            dst[x + y * dst_dim_x] = src[x + y * src_dim_x];
                        else
                            dst[x + y * dst_dim_x] = 0.f;
                    }
                }
            }

            auto do_filtering(fftwf_complex* in, const fftwf_complex* filter,
                              std::uint32_t dim_x, std::uint32_t dim_y) noexcept -> void
            {
                #pragma omp parallel for collapse(2)
                for(auto y = 0u; y < dim_y; ++y)
                {
                    for(auto x = 0u; x < dim_x; ++x)
                    {
                        auto coord = x + y * dim_x;
                        in[coord][0] *= filter[x][0];
                        in[coord][1] *= filter[x][1];
                    }
                }
            }

            auto shrink(const float* src, std::uint32_t src_dim_x,
                              float* dst, std::uint32_t dst_dim_x, std::uint32_t dim_y) noexcept -> void
            {
                #pragma omp parallel for collapse(2)
                for(auto y = 0u; y < dim_y; ++y)
                {
                    for(auto x = 0u; x < dst_dim_x; ++x)
                    {
                        dst[x + y * dst_dim_x] = src[x + y * src_dim_x];
                    }
                }
            }

            auto normalize(float* src, std::uint32_t dim_x, std::uint32_t dim_y, std::uint32_t filter_size) noexcept
                -> void
            {
                #pragma omp parallel for collapse(2)
                for(auto y = 0u; y < dim_y; ++y)
                {
                    for(auto x = 0u; x < dim_x; ++x)
                    {
                        src[x + y * dim_x] /= static_cast<float>(filter_size);
                    }
                }
            }
        }

        auto fftw_deleter::operator()(void* p) noexcept -> void
        {
            fftwf_free(p);
        }

        auto make_filter(std::uint32_t size, float tau) -> filter_buffer_type
        {
            // Note some FFTW quirks: Input initialization has to be done AFTER plan creation,
            // otherwise it will be overwritten
            const auto size_trans = size / 2 + 1;
            const auto n = static_cast<int>(size);

            auto r = make_ptr<float>(size);
            auto k = make_ptr<fftwf_complex>(size_trans);

            auto plan = fftwf_plan_dft_r2c_1d(n, r.get(), k.get(), FFTW_MEASURE | FFTW_PRESERVE_INPUT);

            make_filter_real(r.get(), size, tau);

            fftwf_execute(plan);
            
            #pragma omp parallel for
            for(auto x = 0u; x < size_trans; ++x)
            {
                auto result = tau * std::abs(std::sqrt(std::pow(k[x][0], 2.f) + std::pow(k[x][1], 2.f)));

                k[x][0] = result;
                k[x][1] = result;
            }

            return k;
        }

        auto apply_filter(projection_device_type& p, const filter_buffer_type& k, std::uint32_t filter_size,
                          std::uint32_t n_col) -> void
        {
            // dimensionality of the FFT - 1 in this case
            constexpr auto rank = 1;

            // FFT size for each dimension
            static const auto n = static_cast<int>(filter_size);

            // batched FFT -> set batch size
            static const auto batch = static_cast<int>(n_col);

            // allocate memory for expanded projection (projection width -> filter size)
            thread_local static auto p_exp = make_ptr<float>(filter_size, n_col);

            // allocate memory for transformed projection
            static const auto size_trans = filter_size / 2 + 1;
            thread_local static auto p_trans = make_ptr<fftwf_complex>(size_trans, n_col);

            // set distance between the first elements of two successive lines
            static const auto p_exp_dist = static_cast<int>(filter_size);
            static const auto p_trans_dist = static_cast<int>(size_trans);

            // set distance between two successive elements
            constexpr auto p_exp_stride = 1;
            constexpr auto p_trans_stride = 1;

            // set storage dimensions of data in memory
            static const auto p_exp_nembed = static_cast<int>(p_exp_dist);
            static const auto p_trans_nembed = static_cast<int>(p_trans_dist);

            // create plans for forward and inverse FFT
            thread_local static auto forward = fftwf_plan_many_dft_r2c(rank, &n, batch, p_exp.get(), &p_exp_nembed, p_exp_stride, p_exp_dist, p_trans.get(), &p_trans_nembed, p_trans_stride, p_trans_dist, FFTW_MEASURE | FFTW_PRESERVE_INPUT);

            thread_local static auto inverse = fftwf_plan_many_dft_c2r(rank, &n, batch,
                                                          p_trans.get(), &p_trans_nembed, p_trans_stride, p_trans_dist,
                                                          p_exp.get(), &p_exp_nembed, p_exp_stride, p_exp_dist,
                                                          FFTW_MEASURE | FFTW_DESTROY_INPUT);

            // expand and transform the projection
            expand(p.buf.get(), p.dim_x, p_exp.get(), filter_size, n_col);
            fftwf_execute(forward);

            // apply filter to transformed projection
            do_filtering(p_trans.get(), k.get(), size_trans, n_col);

            // inverse transformation
            fftwf_execute(inverse);

            // shrink to original size and normalize
            shrink(p_exp.get(), filter_size, p.buf.get(), p.dim_x, n_col);
            normalize(p.buf.get(), p.dim_x, p.dim_y, filter_size);
        }
    }
}

