/*
 * @Author: zjj
 * @Date: 2023-12-12 11:21:40
 * @LastEditors: zjj
 * @LastEditTime: 2023-12-13 14:33:33
 * @FilePath: /StitchCuda/include/stitch_cuda_api.cuh
 * @Description:
 *
 * Copyright (c) 2023 by ${git_name_email}, All Rights Reserved.
 */
#pragma once

#include "stitch_cuda.cuh"

namespace ParkingPerception
{
    namespace StitchCuda
    {
        std::shared_ptr<ImgStitch> CreateStitch(std::string config_file);
    } // namespace StitchCuda
} // namespace ParkingPerception