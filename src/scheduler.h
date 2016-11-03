/*
 * This file is part of the ddafa reconstruction program.
 *
 * Copyright (C) 2016 Helmholtz-Zentrum Dresden-Rossendorf
 *
 * Licensed under the EUPL, Version 1.1 or - as soon they will be approved by
 * the European Commission - subsequent version of the EUPL (the "Licence");
 * You may not use this work except in compliance with the Licence.
 * You may obtain a copy of the Licence at:
 *
 * http://ec.europa.eu/idabc/eupl
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the Licence is distributed on an "AS IS" basis,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the Licence for the specific language governing permissions and
 * limitations under the Licence.
 *
 * Date: 28 Oktober 2016
 * Authors: Jan Stephan
 */

#ifndef DDAFA_SCHEDULER_H_
#define DDAFA_SCHEDULER_H_

#include <map>

#include "geometry.h"

namespace ddafa
{
    struct subvolume_info
    {
        subvolume_geometry geo;
        int num;
    };

    /*
     * Based on the target volume dimensions, the detector geometry and the number of projections in the pipeline
     * this function creates subvolume geometries according to the following algorithm:
     *
     * 1) Divide the volume by the number of available devices
     * 2) Check if the subvolume and proj_num projections fit into device memory
     *      a) if yes, return
     *      b) else, divide the volume by 2 until it fits
     */
    auto create_subvolume_information(const volume_geometry& vol_geo, const detector_geometry& det_geo, int proj_num) -> subvolume_info;
}



#endif /* DDAFA_SCHEDULER_H_ */
