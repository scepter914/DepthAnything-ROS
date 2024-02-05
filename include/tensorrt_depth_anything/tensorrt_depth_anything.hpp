// Copyright 2023 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORRT_DEPTH_ANYTHING__TENSORRT_DEPTH_ANYTHING_HPP_
#define TENSORRT_DEPTH_ANYTHING__TENSORRT_DEPTH_ANYTHING_HPP_

#include <cuda_utils/cuda_unique_ptr.hpp>
#include <cuda_utils/stream_unique_ptr.hpp>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <tensorrt_common/tensorrt_common.hpp>
#include <tensorrt_depth_anything/preprocess.hpp>
#include <vector>

namespace tensorrt_depth_anything
{
using cuda_utils::CudaUniquePtr;
using cuda_utils::CudaUniquePtrHost;
using cuda_utils::makeCudaStream;
using cuda_utils::StreamUniquePtr;

/**
 * @class TrtDepth_Anything
 * @brief TensorRT DEPTH_ANYTHING for faster inference
 * @warning Regarding quantization, we recommend use MinMax calibration due to accuracy drop with Entropy calibration.
 */
class TrtDepth_Anything
{
public:
  /**
   * @brief Construct TrtDepth_Anything.
   * @param[in] mode_path ONNX model_path
   * @param[in] precision precision for inference
   * @param[in] build_config configuration including precision, calibration method, DLA, remaining fp16 for first layer, remaining fp16 for last layer and profiler for builder
   * @param[in] use_gpu_preprocess whether use cuda gpu for preprocessing
   * @param[in] calibration_image_list_file path for calibration files (only require for quantization)
   * @param[in] batch_config configuration for batched execution
   * @param[in] max_workspace_size maximum workspace for building TensorRT engine
   */
  TrtDepth_Anything(
    const std::string & model_path, const std::string & precision,
    const tensorrt_common::BuildConfig build_config = tensorrt_common::BuildConfig(),
    const bool use_gpu_preprocess = false, std::string calibration_image_list_file = std::string(),
    const tensorrt_common::BatchConfig & batch_config = {1, 1, 1},
    const size_t max_workspace_size = (1 << 30));
  /**
   * @brief Deconstruct TrtDepth_Anything
   */
  ~TrtDepth_Anything();

  /**
   * @brief run inference including pre-process and post-process
   * @param[out] objects results for object detection
   * @param[in] images batched images
   */
  bool doInference(const std::vector<cv::Mat> & images);

  void initPreprocessBuffer(int width, int height);

  /**
   * @brief output TensorRT profiles for each layer
   */
  void printProfiling(void);

  //cv::Mat getDepthImage(const std::string cFormat, float max_depth);
  cv::Mat getDepthImage();

private:
  /**
   * @brief run preprocess including resizing, letterbox, NHWC2NCHW and toFloat on CPU
   * @param[in] images batching images
   */
  void preprocess(const std::vector<cv::Mat> & images);

  /**
   * @brief run preprocess on GPU
   * @param[in] images batching images
   */
  void preprocessGpu(const std::vector<cv::Mat> & images);

  // bool infer(const std::vector<cv::Mat> & images);
  bool infer();

  std::unique_ptr<tensorrt_common::TrtCommon> trt_common_;

  std::vector<float> input_h_;
  CudaUniquePtr<float[]> input_d_;

  size_t out_elem_num_;
  size_t out_elem_num_per_batch_;
  CudaUniquePtr<float[]> out_prob_d_;

  StreamUniquePtr stream_{makeCudaStream()};

  int batch_size_;
  CudaUniquePtrHost<float[]> out_prob_h_;

  // flag whether preprocess are performed on GPU
  bool use_gpu_preprocess_;
  // host buffer for preprocessing on GPU
  CudaUniquePtrHost<unsigned char[]> image_buf_h_;
  // device buffer for preprocessing on GPU
  CudaUniquePtr<unsigned char[]> image_buf_d_;

  std::vector<int> output_strides_;

  int src_width_;
  int src_height_;

  // host pointer for ROI
  CudaUniquePtrHost<Roi[]> roi_h_;
  // device pointer for ROI
  CudaUniquePtr<Roi[]> roi_d_;

  // flag whether model has multitasks
  int multitask_;
  // buff size for segmentation heads
  CudaUniquePtr<float[]> segmentation_out_prob_d_;
  CudaUniquePtrHost<float[]> segmentation_out_prob_h_;
  size_t segmentation_out_elem_num_;
  size_t segmentation_out_elem_num_per_batch_;
  std::vector<cv::Mat> segmentation_masks_;
  // host buffer for argmax postprocessing on GPU
  CudaUniquePtrHost<unsigned char[]> argmax_buf_h_;
  // device buffer for argmax postprocessing  on GPU
  CudaUniquePtr<unsigned char[]> argmax_buf_d_;
};

}  // namespace tensorrt_depth_anything

#endif  // TENSORRT_DEPTH_ANYTHING__TENSORRT_DEPTH_ANYTHING_HPP_
