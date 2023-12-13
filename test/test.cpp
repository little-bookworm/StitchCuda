#include <opencv2/opencv.hpp>

#include "stitch_cuda_api.cuh"

using namespace ParkingPerception::StitchCuda;

int main()
{
  //构造实例
  std::string config_path = "/hostdata/projects/parking_perception/modules/StitchCuda/config/StitchCuda.yaml";
  std::shared_ptr<ImgStitch> img_stitch = CreateStitch(config_path);

  //初始化
  if (0 != img_stitch->init())
  {
    return 0;
  }

  //准备图像
  std::string front_path = "/hostdata/projects/parking_perception/modules/StitchCuda/test/front.png";
  std::string left_path = "/hostdata/projects/parking_perception/modules/StitchCuda/test/left.png";
  std::string back_path = "/hostdata/projects/parking_perception/modules/StitchCuda/test/back.png";
  std::string right_path = "/hostdata/projects/parking_perception/modules/StitchCuda/test/right.png";
  cv::Mat img_front = cv::imread(front_path);
  cv::Mat img_left = cv::imread(left_path);
  cv::Mat img_back = cv::imread(back_path);
  cv::Mat img_right = cv::imread(right_path);
  std::vector<cv::Mat> raw_images;
  raw_images.emplace_back(img_front);
  raw_images.emplace_back(img_left);
  raw_images.emplace_back(img_back);
  raw_images.emplace_back(img_right);

  //拼接
  if (0 != img_stitch->stitch(raw_images))
  {
    return 0;
  }

  //结果
  cv::Mat output;
  img_stitch->get_result(output);
  std::string save_path = "/hostdata/projects/parking_perception/modules/StitchCuda/test/output.png";
  cv::imwrite(save_path, output);
  std::cout << "Save output img: " << save_path << std::endl;

  return 0;
}