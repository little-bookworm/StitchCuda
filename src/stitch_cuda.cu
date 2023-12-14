#include "stitch_cuda.cuh"

namespace ParkingPerception
{
    namespace StitchCuda
    {
        static __device__ __forceinline__ uchar3 belend(uchar3 a, uchar3 b, float w)
        {
            return make_uchar3(a.x * w + b.x * (1 - w), a.y * w + b.y * (1 - w),
                               a.z * w + b.z * (1 - w));
        }

        static __device__ float bicubic(float x)
        {
            float a = -0.75; // opencv取值，默认为-0.5
            float res = 0.0;
            x = abs(x);
            if (x <= 1.0)
                res = (a + 2) * x * x * x - (a + 3) * x * x + 1;
            else if (x < 2.0)
                res = a * x * x * x - 5 * a * x * x + 8 * a * x - 4 * a;
            return res;
        }

        static __device__ void getImpactFactors(float rowU, float colV, float *rowImFac, float *colImFac, int starti, int startj)
        {
            //取整
            int row = (int)rowU;
            int col = (int)colV;
            float temp;
            //计算行系数因子
            for (int i = 0; i < 4; i++)
            {
                if (starti + i <= 0)
                    temp = rowU - row - (starti + i);
                else
                    temp = (starti + i) - (rowU - row);
                rowImFac[i] = bicubic(temp);
                // printf("i:%d,temp:%.1f,rowImFac:%.1f\n", i, temp, rowImFac[i]);
            }
            //计算列系数因子
            for (int j = 0; j < 4; j++)
            {
                if (startj + j <= 0)
                    temp = colV - col - (startj + j);
                else
                    temp = (startj + j) - (colV - col);
                colImFac[j] = bicubic(temp);
            }
        }

        static __device__ uchar3 bicubic_interpolation(float src_x, float src_y, int width, int height, uint8_t *src, float *rowImFac, float *colImFac)
        {
            //计算outputMat(row,col)在原图中的位置坐标
            float inputrowf = src_y;
            float inputcolf = src_x;
            // printf("inputrowf:%.1f,inputcolf:%.1f\n", inputrowf, inputcolf);
            //取整
            int interow = (int)inputrowf;
            int intecol = (int)inputcolf;
            float row_dy = inputrowf - interow;
            float col_dx = inputcolf - intecol;
            //因为扩展了边界，所以+2
            // interow += 2;
            // intecol += 2;
            int starti = -1, startj = -1;
            //计算行影响因子，列影响因子
            getImpactFactors(inputrowf, inputcolf, rowImFac, colImFac, starti, startj);
            // printf("rowImFac:%.2f,%.2f,%.2f,%.2f\n", rowImFac[0], rowImFac[1], rowImFac[2], rowImFac[3]);
            // printf("colImFac:%.1f,%.1f,%.1f,%.1f\n", colImFac[0], colImFac[1], colImFac[2], colImFac[3]);
            //计算输出图像(row,col)的值
            // Vec3f tempvec(0, 0, 0);
            float3 c_3f = {0, 0, 0};
            for (int i = starti; i < starti + 4; i++)
            {
                for (int j = startj; j < startj + 4; j++)
                {
                    uint8_t *src_ptr = src + (interow + i) * 3 * width + (intecol + j) * 3;
                    float weight = rowImFac[i - starti] * colImFac[j - startj];
                    // if (i == 0 && j == 0)
                    // {
                    //     printf("i:%d,j:%d,weight:%.1f\n", i, j, weight);
                    // }
                    // printf("i:%d,j:%d,weight:%.1f\n", i, j, weight);
                    c_3f.x += src_ptr[0] * weight;
                    c_3f.y += src_ptr[1] * weight;
                    c_3f.z += src_ptr[2] * weight;
                }
            }
            c_3f.x = floorf(c_3f.x + 0.5f); //四舍五入
            c_3f.y = floorf(c_3f.y + 0.5f); //四舍五入
            c_3f.z = floorf(c_3f.z + 0.5f); //四舍五入
            uchar3 c = {uchar(c_3f.x), uchar(c_3f.y), uchar(c_3f.z)};
            // uint8_t *src_ptr = src + interow * 3 * width + intecol * 3;
            // c.x += src_ptr[0];
            // c.y += src_ptr[1];
            // c.z += src_ptr[2];
            // outputMat.at<Vec3f>(row, col) = tempvec;
            return c;
        }

        static __device__ uchar3 bilinear_interpolation(float src_x, float src_y, int width, int height, int line_size, uint8_t fill_value, uint8_t *src, float3 bgr_gain)
        {
            float c0 = fill_value, c1 = fill_value, c2 = fill_value;
            //双线性插值
            if (src_x < -1 || src_x >= width || src_y < -1 || src_y >= height)
            {
                // out of range
                // src_x < -1时，其高位high_x < 0，超出范围
                // src_x >= -1时，其高位high_x >= 0，存在取值
            }
            else
            {
                int y_low = floorf(src_y);
                int x_low = floorf(src_x);
                int y_high = y_low + 1;
                int x_high = x_low + 1;

                uint8_t const_values[] = {fill_value, fill_value, fill_value};
                float ly = src_y - y_low;
                float lx = src_x - x_low;
                float hy = 1 - ly;
                float hx = 1 - lx;
                float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
                uint8_t *v1 = const_values;
                uint8_t *v2 = const_values;
                uint8_t *v3 = const_values;
                uint8_t *v4 = const_values;
                if (y_low >= 0)
                {
                    if (x_low >= 0)
                        v1 = src + y_low * line_size + x_low * 3;

                    if (x_high < width)
                        v2 = src + y_low * line_size + x_high * 3;
                }

                if (y_high < height)
                {
                    if (x_low >= 0)
                        v3 = src + y_high * line_size + x_low * 3;

                    if (x_high < width)
                        v4 = src + y_high * line_size + x_high * 3;
                }

                c0 = floorf((w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0]) * bgr_gain.x + 0.5f); //四舍五入
                c1 = floorf((w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1]) * bgr_gain.y + 0.5f); //四舍五入
                c2 = floorf((w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2]) * bgr_gain.z + 0.5f); //四舍五入
            }
            return make_uchar3(c0, c1, c2);
        }

        static __global__ void stitch_kernel(const float10 *table, int w, int h, ptr4 images, int iw, int ih, uchar3 *output, float3 *bgr_gain, float4 *rowImFac, float4 *colImFac, bool use_bicubic)
        {
            int ix = blockDim.x * blockIdx.x + threadIdx.x; // cuda核索引，也代表目标图像素点位置
            int iy = blockDim.y * blockIdx.y + threadIdx.y; // cuda核索引，也代表目标图像素点位置
            if (ix >= w || iy >= h)
                return;

            int pos = iy * w + ix;     //目标图像素点位置索引
            float10 item = table[pos]; //映射表对应位置的元素: flag,weight,front_x,front_y,left_x,left_y,back_x,back_y,right_x,right_y
            int flag = item.x[0];      // 0:front,1:left,2:back,3:right,4:(back,left),5:(front,right),6:(front,left),7;(back,right)
            float weight = item.x[1];  //融合权值

            if (flag == -1)
                return;
            if (flag < 4) //单副图像
            {
                float x = item.x[2 + flag * 2 + 0];
                float y = item.x[2 + flag * 2 + 1];
                if (use_bicubic) //使用双立方插值或双线性插值
                {
                    output[pos] = bicubic_interpolation(x, y, iw, ih, (uint8_t *)images.v[flag], (float *)(rowImFac + iy * w + ix), (float *)(colImFac + iy * w + ix));
                }
                else
                {
                    output[pos] = bilinear_interpolation(x, y, iw, ih, 3 * iw, 0, (uint8_t *)images.v[flag], bgr_gain[flag]);
                }
            }
            else
            {
                const int idxs[][2] = {{2, 1}, {0, 3}, {0, 1}, {2, 3}};
                int a = idxs[flag - 4][0];        //第一幅源图像索引
                int b = idxs[flag - 4][1];        //第二幅源图像索引
                float ax = item.x[2 + a * 2 + 0]; //第一幅源图像坐标x
                float ay = item.x[2 + a * 2 + 1]; //第一幅源图像坐标y
                float bx = item.x[2 + b * 2 + 0]; //第二幅源图像坐标x
                float by = item.x[2 + b * 2 + 1]; //第二幅源图像坐标y
                uchar3 pixel_a, pixel_b;
                if (use_bicubic) //使用双立方插值或双线性插值
                {
                    pixel_a = bicubic_interpolation(ax, ay, iw, ih, (uint8_t *)images.v[a], (float *)(rowImFac + iy * w + ix), (float *)(colImFac + iy * w + ix));
                    pixel_b = bicubic_interpolation(bx, by, iw, ih, (uint8_t *)images.v[b], (float *)(rowImFac + iy * w + ix), (float *)(colImFac + iy * w + ix));
                }
                else
                {
                    pixel_a = bilinear_interpolation(ax, ay, iw, ih, 3 * iw, 0, (uint8_t *)images.v[a], bgr_gain[a]);
                    pixel_b = bilinear_interpolation(bx, by, iw, ih, 3 * iw, 0, (uint8_t *)images.v[b], bgr_gain[b]);
                }
                output[pos] = belend(pixel_a, pixel_b, weight); //两幅图融合
            }
        }

        static __global__ void cal_sum_kernel(float3 *Para, int N, float3 *blocksum_cuda)
        {
            //计算线程ID号
            // blockIdx.x为线程块的ID号
            // blockDim.x每个线程块中包含的线程总个数
            // threadIdx.x为每个线程块中的线程ID号
            int tid = blockIdx.x * blockDim.x + threadIdx.x;

            if (tid < N)
            {
                for (int index = 1; index < blockDim.x; index = (index * 2))
                {
                    if (threadIdx.x % (index * 2) == 0)
                    {
                        Para[tid].x += Para[tid + index].x; //规约求和
                        Para[tid].y += Para[tid + index].y; //规约求和
                        Para[tid].z += Para[tid + index].z; //规约求和
                    }

                    __syncthreads(); //同步线程块中的所有线程
                }

                if (threadIdx.x == 0) //整个数组相加完成后，将共享内存数组0号元素的值赋给全局内存数组0号元素，最后返回CPU端
                {
                    blocksum_cuda[blockIdx.x].x = Para[tid].x;
                    blocksum_cuda[blockIdx.x].y = Para[tid].y;
                    blocksum_cuda[blockIdx.x].z = Para[tid].z;
                    // printf("i: %d,x:%f,y:%f,z:%f\n", blockIdx.x, blocksum_cuda[blockIdx.x].x, blocksum_cuda[blockIdx.x].y, blocksum_cuda[blockIdx.x].z);
                }
            }
        }

        ImgStitch::ImgStitch(std::string config_path) : config_path_(config_path)
        {
        }

        ImgStitch::~ImgStitch()
        {
            destroy();
        }

        int ImgStitch::init()
        {
            //读取配置
            if (0 != load_config())
            {
                std::cout << "[ImgStitch]->[init] Failed to load config file." << std::endl;
                return -1;
            }

            //打开并判断映射表可用性
            FILE *f = fopen(table_path_.c_str(), "rb");
            if (f == nullptr)
            {
                std::cout << "[ImgStitch]->[init] Failed to load table: " << table_path_ << std::endl;
                return -1;
            }

            fseek(f, 0, SEEK_END);
            size_t size = ftell(f);
            fseek(f, 0, SEEK_SET);

            if (size != w_ * h_ * 10 * sizeof(float))
            {
                std::cout << "[ImgStitch]->[init] Invalid table file: " << table_path_ << std::endl;
                fclose(f);
                return -1;
            }

            // 设置device
            checkRuntime(cudaSetDevice(0));

            // 设置stream;
            checkRuntime(cudaStreamCreate(&stream));

            //读取映射表数据
            unsigned char *table_host = new unsigned char[size];
            fread(table_host, 1, size, f);
            fclose(f);

            //拼接图cpu分配内存
            output_.create(h_, w_, CV_8UC3);

            //原图cuda分配内存
            images_device_.clear();
            for (int i = 0; i < numcam_; ++i)
            {
                unsigned char *device_ptr = nullptr;
                checkRuntime(cudaMalloc(&device_ptr, camw_ * camh_ * 3 * sizeof(unsigned char)));
                images_device_.push_back(device_ptr);
            }

            //拼接图cuda分配内存
            checkRuntime(cudaMalloc(&output_view_, w_ * h_ * 3 * sizeof(unsigned char)));

            //映射表cuda分配内存
            checkRuntime(cudaMalloc(&table_, size));

            //映射表数据拷贝至cuda
            checkRuntime(cudaMemcpy(table_, table_host, size, cudaMemcpyHostToDevice));
            delete[] table_host;

            //白平衡初始化
            if (use_lum_banlance)
            {
                N = camw_ * camh_;                                                      //待求和数据量
                sumblock_x = 1024;                                                      //每个线程块的线程数
                sumgrid_x = (N % sumblock_x) ? (N / sumblock_x + 1) : (N / sumblock_x); //分配的cuda block数量
                blocksum_host = new float3[sumgrid_x];                                  // cpu上分配线程块数量的内存
                checkRuntime(cudaMalloc(&blocksum_cuda, sizeof(float3) * sumgrid_x));   // gpu上分配线程块数量的内存
                images_float_device_.clear();
                for (int i = 0; i < numcam_; ++i) // gpu分配float图像内存
                {
                    float *device_float_ptr = nullptr;
                    checkRuntime(cudaMalloc(&device_float_ptr, camw_ * camh_ * sizeof(float3)));
                    images_float_device_.push_back(device_float_ptr);
                }
            }
            bgr_gain_host = new float3[4]; // cpu上存储的bgr缩放系数
            for (int i = 0; i < 4; ++i)    //初始化为1
            {
                bgr_gain_host[i].z = 1.0;
                bgr_gain_host[i].y = 1.0;
                bgr_gain_host[i].x = 1.0;
            }
            checkRuntime(cudaMalloc(&bgr_gain_device, sizeof(float3) * 4)); // gpu上分配bgr_gain内存

            //双立方插值初始化
            if (use_bicubic)
            {
                checkRuntime(cudaMalloc(&rowImFac_device, sizeof(float4) * w_ * h_)); //双立方插值影响因子
                checkRuntime(cudaMalloc(&colImFac_device, sizeof(float4) * w_ * h_)); //双立方插值影响因子
            }

            std::cout << "[ImgStitch]->[init] Init success." << std::endl;

            return 0;
        }

        int ImgStitch::stitch(const std::vector<cv::Mat> &images)
        {
            //判断原图数量
            if (images.size() != numcam_)
            {
                std::cout << "[ImgStitch]->[stitch] Unsupported image number!!!" << std::endl;
                return -1;
            }

            //原图cpu数据拷贝至gpu
            for (int i = 0; i < images.size(); ++i)
            {
                auto &image = images[i];
                if (image.cols != camw_ || image.rows != camh_)
                {
                    std::cout << "[ImgStitch]->[stitch] Invalid image size: " << image.cols << "," << image.rows << std::endl;
                    return -1;
                }

                checkRuntime(cudaMemcpyAsync(images_device_[i], image.data,
                                             image.cols * image.rows * 3 * sizeof(unsigned char),
                                             cudaMemcpyHostToDevice, stream));
            }

            //白平衡
            if (use_lum_banlance)
            {
                // float类型图片
                cv::Mat img_front_float;                                    // float类型图片
                images[0].clone().convertTo(img_front_float, CV_32F, 1, 0); // 1、0分别是比例因子，y = a*x + b
                cv::Mat img_left_float;                                     // float类型图片
                images[1].clone().convertTo(img_left_float, CV_32F, 1, 0);  // 1、0分别是比例因子，y = a*x + b
                cv::Mat img_back_float;                                     // float类型图片
                images[2].clone().convertTo(img_back_float, CV_32F, 1, 0);  // 1、0分别是比例因子，y = a*x + b
                cv::Mat img_right_float;                                    // float类型图片
                images[3].clone().convertTo(img_right_float, CV_32F, 1, 0); // 1、0分别是比例因子，y = a*x + b
                std::vector<cv::Mat> float_images;
                float_images.emplace_back(img_front_float);
                float_images.emplace_back(img_left_float);
                float_images.emplace_back(img_back_float);
                float_images.emplace_back(img_right_float);

                if (0 != awb_and_lum_banlance(float_images))
                {
                    std::cout << "[ImgStitch]->[stitch] Failed to awb_and_lum_banlance!!!" << std::endl;
                    return -1;
                }
            }
            checkRuntime(cudaMemcpy(bgr_gain_device, bgr_gain_host, sizeof(float3) * 4, cudaMemcpyHostToDevice)); // bgr_gain从cpu拷贝至gpu

            //拼接
            ptr4 images_ptr;
            memcpy(images_ptr.v, images_device_.data(), sizeof(images_device_[0]) * 4);
            dim3 block(32, 32);
            dim3 grid((w_ + block.x - 1) / block.x, (h_ + block.y - 1) / block.y);
            stitch_kernel<<<grid, block, 0, stream>>>(
                table_, w_, h_, images_ptr, camw_, camh_, (uchar3 *)output_view_, bgr_gain_device, rowImFac_device, colImFac_device, use_bicubic);

            checkRuntime(cudaMemcpyAsync(output_.data, output_view_,
                                         output_.rows * output_.cols * 3 * sizeof(unsigned char),
                                         cudaMemcpyDeviceToHost, stream));
            checkRuntime(cudaStreamSynchronize(stream));

            return 0;
        }

        void ImgStitch::get_result(cv::Mat &out)
        {
            out = output_.clone();
        }

        int ImgStitch::load_config()
        {
            //导入yaml文件
            YAML::Node config;
            try
            {
                config = YAML::LoadFile(config_path_);
            }
            catch (const std::exception &e)
            {
                std::cout << "[ImgStitch]->[load_config] No config file: " << config_path_ << std::endl;
                return -1;
            }

            //导入配置参数
            auto img_params = config["img_params"];
            w_ = img_params["stitch_img"]["w"].as<int>();
            h_ = img_params["stitch_img"]["h"].as<int>();
            camw_ = img_params["raw_img"]["w"].as<int>();
            camh_ = img_params["raw_img"]["h"].as<int>();
            numcam_ = img_params["raw_img"]["num"].as<int>();
            if (w_ == 0 || h_ == 0 || camw_ == 0 || camh_ == 0 || numcam_ == 0)
            {
                std::cout << "[ImgStitch]->[load_config] img_params error!!!" << std::endl;
                return -1;
            }
            table_path_ = config["table_path"].as<std::string>();
            if (table_path_ == "")
            {
                std::cout << "[ImgStitch]->[load_config] table_path is empty!!!" << std::endl;
                return -1;
            }
            use_lum_banlance = config["use_lum_banlance"].as<bool>();
            use_bicubic = config["use_bicubic"].as<bool>();

            return 0;
        }

        int ImgStitch::awb_and_lum_banlance(const std::vector<cv::Mat> &float_images)
        {
            //判断原图数量
            if (float_images.size() != numcam_)
            {
                std::cout << "[ImgStitch]->[awb_and_lum_banlance] Unsupported float_images number!!!" << std::endl;
                return -1;
            }

            // float cpu数据拷贝至gpu
            for (int i = 0; i < float_images.size(); ++i)
            {
                auto &float_image = float_images[i];
                if (float_image.cols != camw_ || float_image.rows != camh_)
                {
                    std::cout << "[ImgStitch]->[awb_and_lum_banlance] Invalid float_image size: " << float_image.cols << "," << float_image.rows << std::endl;
                    return -1;
                }

                checkRuntime(cudaMemcpyAsync(images_float_device_[i], float_image.data,
                                             float_image.cols * float_image.rows * sizeof(float3),
                                             cudaMemcpyHostToDevice, stream));
            }

            checkRuntime(cudaStreamSynchronize(stream));
            dim3 sumblock(sumblock_x);                                                  //设置每个线程块有1024个线程
            dim3 sumgrid(((N % sumblock_x) ? (N / sumblock_x + 1) : (N / sumblock_x))); //设置总共有多少个线程块
            float3 sts[4];                                                              //存储4组brg均值
            float gray[4] = {0, 0, 0, 0};                                               //存储4组灰度值
            float gray_ave = 0;                                                         //灰度平均值
            for (int i = 0; i < 4; ++i)                                                 //单张图做规约求和
            {
                cal_sum_kernel<<<sumgrid, sumblock>>>((float3 *)images_float_device_[i], N, blocksum_cuda);                 //单张图rgb累加至每个block上
                checkRuntime(cudaMemcpy(blocksum_host, blocksum_cuda, sizeof(float3) * sumgrid_x, cudaMemcpyDeviceToHost)); // block上的数据拷贝至cpu上

                sts[i] = {0, 0, 0};
                for (int j = 0; j < sumgrid_x; j++) //在CPU端对所有线程块的规约求和结果做串行求和
                {
                    sts[i].x += blocksum_host[j].x;
                    sts[i].y += blocksum_host[j].y;
                    sts[i].z += blocksum_host[j].z;
                }
                sts[i].x /= N;                                                    // b均值
                sts[i].y /= N;                                                    // g均值
                sts[i].z /= N;                                                    // r均值
                gray[i] = sts[i].x * 0.114 + sts[i].y * 0.587 + sts[i].z * 0.299; // 灰度值
                gray_ave += gray[i];                                              // 4张图灰度值累加
            }
            gray_ave /= 4;              // 4张图平均灰度值
            for (int i = 0; i < 4; ++i) //这里没看明白
            {
                float lum_gain = gray_ave / gray[i];
                bgr_gain_host[i].z = sts[i].y * lum_gain / sts[i].z;
                bgr_gain_host[i].y = lum_gain;
                bgr_gain_host[i].x = sts[i].y * lum_gain / sts[i].x;
                // printf("gains_rgb : %f | %f | %f\n", bgr_gain_host[i].z, bgr_gain_host[i].y, bgr_gain_host[i].x);
            }

            return 0;
        }

        void ImgStitch::destroy()
        {
            if (stream)
            {
                checkRuntime(cudaStreamDestroy(stream));
                stream = nullptr;
            }

            for (int i = 0; i < images_device_.size(); ++i)
            {
                if (images_device_[i])
                {
                    checkRuntime(cudaFree(images_device_[i]));
                }
            }
            images_device_.clear();

            if (table_)
            {
                checkRuntime(cudaFree(table_));
                table_ = nullptr;
            }

            if (output_view_)
            {
                checkRuntime(cudaFree(output_view_));
                output_view_ = nullptr;
            }

            if (use_lum_banlance)
            {
                if (blocksum_host)
                {
                    delete[] blocksum_host;
                    blocksum_host = nullptr;
                }

                if (blocksum_cuda)
                {
                    checkRuntime(cudaFree(blocksum_cuda));
                    blocksum_cuda = nullptr;
                }

                for (int i = 0; i < images_float_device_.size(); ++i)
                {
                    if (images_float_device_[i])
                    {
                        checkRuntime(cudaFree(images_float_device_[i]));
                    }
                }
                images_float_device_.clear();
            }

            if (bgr_gain_host)
            {
                delete[] bgr_gain_host;
                bgr_gain_host = nullptr;
            }

            if (bgr_gain_device)
            {
                checkRuntime(cudaFree(bgr_gain_device));
                bgr_gain_device = nullptr;
            }

            if (use_bicubic)
            {
                if (rowImFac_device)
                {
                    checkRuntime(cudaFree(rowImFac_device));
                    rowImFac_device = nullptr;
                }
                if (colImFac_device)
                {
                    checkRuntime(cudaFree(colImFac_device));
                    colImFac_device = nullptr;
                }
            }
        }
    }
}
