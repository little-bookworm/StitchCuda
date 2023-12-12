#include "stitch_cuda_api.cuh"

namespace ParkingPerception
{
namespace StitchCuda
{
    ImgStitch *CreateStitch(std::string config_file)
    {
        return new ImgStitch(config_file);
    }
} // namespace StitchCuda
} // namespace ParkingPerception