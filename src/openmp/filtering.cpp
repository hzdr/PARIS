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
 * Date: 05 December 2016
 * Authors: Jan Stephan <j.stephan@hzdr.de>
 */

#include <cmath>
#include <cstdint>
#include <memory>
#include <numeric>

#include <fftw3.h>

#include "backend.h"

namespace ddafa
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
            auto make_filter_real(std::uint32_t size, float tau) -> device_ptr_1D<float>
            {
                auto js = make_device_ptr<std::int32_t>(size);
                auto size_i = static_cast<std::int32_t>(size);
                auto j = -(size - 2) / 2;
                std::iota(js.get(), js.get() + size, j);

                auto r = std::make_device_ptr<float>(size);

                #pragma omp parallel for
                for(auto x = 0u; x < size; ++x)
                {
                    if(j[x] == 0)
                        r[x] = (1.f / 8.f) * (1.f / std::pow(tau, 2.f));
                    else
                    {
                        if(j[x] % 2 == 0)
                            r[x] = 0.f;
                        else
                            r[x] = -(1.f / (2.f * std::pow(j[x], 2.f)
                                   * std::pow(M_PI, 2.f)
                                   * std::pow(tau, 2.f)));
                    }
                }
            }   
        }

        auto make_filter(std::uint32_t size, float tau) -> device_ptr_1D<fft::complex_type>
        {
            auto r = make_filter_real(size, tau);

            auto size_trans = size / 2 + 1;
            auto k = make_device_ptr<fft::complex_type>(size_trans);

            auto n = static_cast<int>(size);

            auto plan = fftwf_plan_dft_r2c_1d(n, r.get(), k.get(), FFTW_MEASURE | FFTW_PRESERVE_INPUT);
            fftwf_execute(plan);
            
            #pragma omp parallel for
            for(auto x = 0u; x < size_trans; ++x)
            {
                auto result = tau * std::abs(std::sqrt(std::pow(k[x][0], 2.f) + std::powf(k[x][1], 2.f)));

                k[x][0] = result;
                k[x][1] = result;
            }
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

