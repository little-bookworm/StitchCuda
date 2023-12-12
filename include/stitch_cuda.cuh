/*
 * @Author: zjj
 * @Date: 2023-12-12 09:33:43
 * @LastEditors: zjj
 * @LastEditTime: 2023-12-12 17:52:29
 * @FilePath: /StitchCuda/include/stitch_cuda.cuh
 * @Description:
 *
 * Copyright (c) 2023 by ${git_name_email}, All Rights Reserved.
 */
#pragma once

#include <iostream>
#include <string>
#include <stdio.h>
#include <cuda_runtime.h>
#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>

#include "common.cuh"

namespace ParkingPerception
{
    namespace StitchCuda
    {
        struct float10
        {
            float x[10];
        };

        struct ptr4
        {
            uchar3 *v[4];
        };

        static __device__ __forceinline__ uchar3 belend(uchar3 a, uchar3 b, float w);

        static __device__ float bicubic(float x);

        static __device__ void getImpactFactors(float rowU, float colV, float *rowImFac, float *colImFac, int starti,
                                                int startj);

        static __device__ uchar3 bicubic_interpolation(float src_x, float src_y, int width, int height, uint8_t *src,
                                                       float *rowImFac, float *colImFac);

        static __device__ uchar3 bilinear_interpolation(float src_x, float src_y, int width, int height, int line_size,
                                                        uint8_t fill_value, uint8_t *src, float3 bgr_gain);

        static __global__ void stitch_kernel(const float10 *table, int w, int h, ptr4 images, int iw, int ih, uchar3 *output,
                                             float3 *bgr_gain, float4 *rowImFac, float4 *colImFac, bool use_bicubic);

        static __global__ void cal_sum_kernel(float3 *Para, int N, float3 *blocksum_cuda);

        class ImgStitch
        {
        public:
            ImgStitch(std::string config_path);
            ~ImgStitch();
            int init();
            int stitch(const std::vector<cv::Mat> &images, const std::vector<cv::Mat> &float_images);
            void get_result(cv::Mat &out);

        private:
            int load_config(std::string &config_path);
            int awb_and_lum_banlance(const std::vector<cv::Mat> &float_images);
            void destroy();

        private:
            // cuda
            cudaStream_t stream;
            //拼接图
            cv::Mat output_;                       //拼接图
            int w_ = 0;                            //拼接图尺寸
            int h_ = 0;                            //拼接图尺寸
            unsigned char *output_view_ = nullptr; //拼接图cuda地址
            //原图
            int camw_ = 0;                               //原图尺寸
            int camh_ = 0;                               //原图尺寸
            int numcam_ = 0;                             //原图数量
            std::vector<unsigned char *> images_device_; //原图cuda地址
            //映射表
            std::string table_path_ = ""; //映射表路径
            float10 *table_ = nullptr;    //映射表cuda地址
            //白平衡
            bool use_lum_banlance = false;             //是否使用白平衡
            int N;                                     //单张图像像素数量
            int sumblock_x;                            //每个线程块的线程数
            int sumgrid_x;                             //分配的cuda block数量
            float3 *blocksum_cuda;                     // cuda block上存储的累加值
            float3 *blocksum_host;                     // cpu上的block累加值
            std::vector<float *> images_float_device_; //存储float类型mat的vector
            float3 *bgr_gain_host;                     // cpu上存储的bgr缩放系数
            float3 *bgr_gain_device;                   // gpu上存储的bgr缩放系数
            //双立方插值
            bool use_bicubic = false; //是否使用双立方插值
            float4 *rowImFac_device;  // gpu上存储的行影响因子
            float4 *colImFac_device;  // gpu上存储的列影响因子
        };

    }
}