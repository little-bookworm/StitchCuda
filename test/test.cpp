#include <opencv2/opencv.hpp>

#include "stitch_cuda_api.cuh"

using namespace ParkingPerception::StitchCuda;

int main()
{
  //构造实例
  std::string config_path = "/hostdata/projects/parking_perception/modules/StitchCuda/config/StitchCuda.yaml";
  ImgStitch* img_stitch = CreateStitch(config_path);

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
  std::vector<cv::Mat> float_images;
  cv::Mat img_front_float;                                     // float类型图片
  img_front.clone().convertTo(img_front_float, CV_32F, 1, 0);  // 1、0分别是比例因子，y = a*x + b
  cv::Mat img_left_float;                                      // float类型图片
  img_left.clone().convertTo(img_left_float, CV_32F, 1, 0);    // 1、0分别是比例因子，y = a*x + b
  cv::Mat img_back_float;                                      // float类型图片
  img_back.clone().convertTo(img_back_float, CV_32F, 1, 0);    // 1、0分别是比例因子，y = a*x + b
  cv::Mat img_right_float;                                     // float类型图片
  img_right.clone().convertTo(img_right_float, CV_32F, 1, 0);  // 1、0分别是比例因子，y = a*x + b
  float_images.emplace_back(img_front_float);
  float_images.emplace_back(img_left_float);
  float_images.emplace_back(img_back_float);
  float_images.emplace_back(img_right_float);

  //拼接
  if (0 != img_stitch->stitch(raw_images, float_images))
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