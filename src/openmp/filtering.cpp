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
        namespace fft
        {
            auto make_forward_plan(int rank, int* n, int batch_size,
                                float* in, int* inembed, int istride, int idist,
                                complex_type* out, int* onembed, int ostride, int odist) -> forward_plan_type
            {
                return fftwf_plan_many_dft_r2c(rank, n, batch_size,
                                              in, inembed, istride, idist,
                                              out, onembed, ostride, odist,
                                              FFTW_MEASURE | FFTW_PRESERVE_INPUT);
            }

            auto make_inverse_plan(int rank, int* n, int batch_size,
                                complex_type* in, int* inembed, int istride, int idist,
                                float* out, int* onembed, int ostride, int odist) -> inverse_plan_type
            {
                return fftwf_plan_many_dft_c2r(rank, n, batch_size,
                                              in, inembed, istride, idist,
                                              out, onembed, ostride, odist,
                                              FFTW_MEASURE | FFTW_DESTROY_INPUT);
            }
        }

        namespace
        {
            auto make_filter_real(fft::pointer<fft::real_type>& r, std::uint32_t size, float tau) -> void
            {
                auto js = make_device_ptr<std::int32_t>(size);
                auto j = -(static_cast<std::int32_t>(size) - 2) / 2;
                std::iota(js.get(), js.get() + size, j);

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
                            r[x] = -(1.f / (2.f * std::pow(static_cast<fft::real_type>(js[x]), 2.f)
                                   * std::pow(static_cast<fft::real_type>(M_PI), 2.f)
                                   * std::pow(tau, 2.f)));
                    }
                }
            }   
        }

        auto make_filter(std::uint32_t size, float tau) -> fft::pointer<fft::complex_type>
        {
            // Note some FFTW quirks: Input initialization has to be done AFTER plan creation,
            // otherwise it will be overwritten
            auto size_trans = size / 2 + 1;
            auto n = static_cast<int>(size);

            auto r = fft::make_ptr<fft::real_type>(size);
            auto k = fft::make_ptr<fft::complex_type>(size_trans);

            auto plan = fftwf_plan_dft_r2c_1d(n, r.get(), k.get(), FFTW_MEASURE | FFTW_PRESERVE_INPUT);

            make_filter_real(r, size, tau);

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
    
        namespace detail
        {
            auto do_filtering(fft::complex_type* in, const fft::complex_type* filter,
                              std::uint32_t dim_x, std::uint32_t dim_y) -> void
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
        }
    }
}

