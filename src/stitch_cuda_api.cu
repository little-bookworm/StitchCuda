#include "stitch_cuda_api.cuh"

namespace ParkingPerception
{
    namespace StitchCuda
    {
        std::shared_ptr<ImgStitch> CreateStitch(std::string config_file)
        {
            return std::make_shared<ImgStitch>(config_file);
        }
    } // namespace StitchCuda
} // namespace ParkingPerception