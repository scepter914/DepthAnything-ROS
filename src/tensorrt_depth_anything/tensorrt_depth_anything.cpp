#include <algorithm>
#include <filesystem>
#include <functional>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tensorrt_depth_anything/calibrator.hpp>
#include <tensorrt_depth_anything/preprocess.hpp>
#include <tensorrt_depth_anything/tensorrt_depth_anything.hpp>
#include <vector>

#include "cuda_utils/cuda_check_error.hpp"
#include "cuda_utils/cuda_unique_ptr.hpp"

namespace
{
namespace fs = std::filesystem;

static void trimLeft(std::string & s)
{
  s.erase(s.begin(), find_if(s.begin(), s.end(), [](int ch) { return !isspace(ch); }));
}

static void trimRight(std::string & s)
{
  s.erase(find_if(s.rbegin(), s.rend(), [](int ch) { return !isspace(ch); }).base(), s.end());
}

static std::string trim(std::string & s)
{
  trimLeft(s);
  trimRight(s);
  return s;
}

static bool fileExists(const std::string & file_name, bool verbose)
{
  if (!std::filesystem::exists(std::filesystem::path(file_name))) {
    if (verbose) {
      std::cout << "File does not exist : " << file_name << std::endl;
    }
    return false;
  }
  return true;
}

static std::vector<std::string> loadListFromTextFile(const std::string & filename)
{
  assert(fileExists(filename, true));
  std::vector<std::string> list;

  std::ifstream f(filename);
  if (!f) {
    std::cout << "failed to open " << filename << std::endl;
    assert(0);
  }

  std::string line;
  while (std::getline(f, line)) {
    if (line.empty()) {
      continue;
    } else {
      list.push_back(trim(line));
    }
  }

  return list;
}

static std::vector<std::string> loadImageList(
  const std::string & filename, const std::string & prefix)
{
  std::vector<std::string> fileList = loadListFromTextFile(filename);
  for (auto & file : fileList) {
    if (fileExists(file, false)) {
      continue;
    } else {
      std::string prefixed = prefix + file;
      if (fileExists(prefixed, false))
        file = prefixed;
      else
        std::cerr << "WARNING: couldn't find: " << prefixed << " while loading: " << filename
                  << std::endl;
    }
  }
  return fileList;
}
}  // anonymous namespace

namespace tensorrt_depth_anything
{
TrtDepth_Anything::TrtDepth_Anything(
  const std::string & model_path, const std::string & precision,
  tensorrt_common::BuildConfig build_config, const bool use_gpu_preprocess,
  std::string calibration_image_list_path, const tensorrt_common::BatchConfig & batch_config,
  const size_t max_workspace_size)
{
  //Todo : use_gpu_preprocess
  src_width_ = -1;
  src_height_ = -1;
  batch_size_ = batch_config[2];
  if (precision == "int8") {
    if (build_config.clip_value <= 0.0) {
      if (calibration_image_list_path.empty()) {
        throw std::runtime_error(
          "calibration_image_list_path should be passed to generate int8 engine "
          "or specify values larger than zero to clip_value.");
      }
    } else {
      // if clip value is larger than zero, calibration file is not needed
      calibration_image_list_path = "";
    }

    int max_batch_size = batch_size_;
    nvinfer1::Dims input_dims = tensorrt_common::get_input_dims(model_path);
    std::vector<std::string> calibration_images;
    if (calibration_image_list_path != "") {
      calibration_images = loadImageList(calibration_image_list_path, "");
    }
    tensorrt_depth_anything::ImageStream stream(max_batch_size, input_dims, calibration_images);
    fs::path calibration_table{model_path};
    std::string calibName = "";
    std::string ext = "";
    if (build_config.calib_type_str == "Entropy") {
      ext = "EntropyV2-";
    } else if (
      build_config.calib_type_str == "Legacy" || build_config.calib_type_str == "Percentile") {
      ext = "Legacy-";
    } else {
      ext = "MinMax-";
    }

    ext += "calibration.table";
    calibration_table.replace_extension(ext);
    fs::path histogram_table{model_path};
    ext = "histogram.table";
    histogram_table.replace_extension(ext);

    std::unique_ptr<nvinfer1::IInt8Calibrator> calibrator;
    if (build_config.calib_type_str == "Entropy") {
      calibrator.reset(
        new tensorrt_depth_anything::Int8EntropyCalibrator(stream, calibration_table));

    } else if (
      build_config.calib_type_str == "Legacy" || build_config.calib_type_str == "Percentile") {
      const double quantile = 0.999999;
      const double cutoff = 0.999999;
      calibrator.reset(new tensorrt_depth_anything::Int8LegacyCalibrator(
        stream, calibration_table, histogram_table, true, quantile, cutoff));
    } else {
      calibrator.reset(
        new tensorrt_depth_anything::Int8MinMaxCalibrator(stream, calibration_table));
    }

    trt_common_ = std::make_unique<tensorrt_common::TrtCommon>(
      model_path, precision, std::move(calibrator), batch_config, max_workspace_size, build_config);
  } else {
    trt_common_ = std::make_unique<tensorrt_common::TrtCommon>(
      model_path, precision, nullptr, batch_config, max_workspace_size, build_config);
  }
  trt_common_->setup();

  if (use_gpu_preprocess) {
    use_gpu_preprocess_ = true;
    image_buf_h_ = nullptr;
    image_buf_d_ = nullptr;
  } else {
    use_gpu_preprocess_ = false;
  }

  if (!trt_common_->isInitialized()) {
    return;
  }

  // GPU memory allocation
  const auto input_dims = trt_common_->getBindingDimensions(0);
  const auto input_size =
    std::accumulate(input_dims.d + 1, input_dims.d + input_dims.nbDims, 1, std::multiplies<int>());

  const auto output_dims = trt_common_->getBindingDimensions(1);
  input_d_ = cuda_utils::make_unique<float[]>(batch_config[2] * input_size);
  out_elem_num_ = std::accumulate(
    output_dims.d + 1, output_dims.d + output_dims.nbDims, 1, std::multiplies<int>());
  out_elem_num_ = out_elem_num_ * batch_config[2];
  out_prob_d_ = cuda_utils::make_unique<float[]>(out_elem_num_);
  out_prob_h_ = cuda_utils::make_unique_host<float[]>(out_elem_num_, cudaHostAllocPortable);
}

TrtDepth_Anything::~TrtDepth_Anything()
{
  if (use_gpu_preprocess_) {
    if (image_buf_h_) {
      image_buf_h_.reset();
    }
    if (image_buf_d_) {
      image_buf_d_.reset();
    }
    if (argmax_buf_d_) {
      argmax_buf_d_.reset();
    }
  }
}

void TrtDepth_Anything::initPreprocessBuffer(int width, int height)
{
  // if size of source input has been changed...
  if (src_width_ != -1 || src_height_ != -1) {
    if (width != src_width_ || height != src_height_) {
      // Free cuda memory to reallocate
      if (image_buf_h_) {
        image_buf_h_.reset();
      }
      if (image_buf_d_) {
        image_buf_d_.reset();
      }
    }
  }

  src_width_ = width;
  src_height_ = height;
  if (use_gpu_preprocess_) {
    auto input_dims = trt_common_->getBindingDimensions(0);
    bool const hasRuntimeDim = std::any_of(
      input_dims.d, input_dims.d + input_dims.nbDims,
      [](int32_t input_dim) { return input_dim == -1; });
    if (hasRuntimeDim) {
      input_dims.d[0] = batch_size_;
    }
    if (!image_buf_h_) {
      trt_common_->setBindingDimensions(0, input_dims);
    }
    if (!image_buf_h_) {
      image_buf_h_ = cuda_utils::make_unique_host<unsigned char[]>(
        width * height * 3 * batch_size_, cudaHostAllocWriteCombined);
      image_buf_d_ = cuda_utils::make_unique<unsigned char[]>(width * height * 3 * batch_size_);
    }
  }
}

void TrtDepth_Anything::printProfiling(void) { trt_common_->printProfiling(); }

void TrtDepth_Anything::preprocess(const std::vector<cv::Mat> & images)
{
  const auto batch_size = images.size();
  auto input_dims = trt_common_->getBindingDimensions(0);
  input_dims.d[0] = batch_size;
  trt_common_->setBindingDimensions(0, input_dims);
  const float input_chan = static_cast<float>(input_dims.d[1]);
  const float input_height = static_cast<float>(input_dims.d[2]);
  const float input_width = static_cast<float>(input_dims.d[3]);
  std::vector<cv::Mat> dst_images;

  for (const auto & image : images) {
    cv::Mat dst_image;
    const auto scale_size = cv::Size(input_width, input_height);
    cv::resize(image, dst_image, scale_size, 0, 0, cv::INTER_CUBIC);
    dst_images.emplace_back(dst_image);
  }
  int volume = batch_size * input_chan * input_height * input_width;
  input_h_.resize(volume);
  std::vector<float> mean{0.485f, 0.456f, 0.406f};
  std::vector<float> std{0.229f, 0.224f, 0.225f};
  const size_t strides_cv[4] = {
    static_cast<size_t>(input_width * input_chan * input_height),
    static_cast<size_t>(input_width * input_chan), static_cast<size_t>(input_chan), 1};
  //NCHW
  const size_t strides[4] = {
    static_cast<size_t>(input_height * input_width * input_chan),
    static_cast<size_t>(input_height * input_width), static_cast<size_t>(input_width), 1};
  for (int n = 0; n < (int)batch_size; n++) {
    for (int h = 0; h < input_height; h++) {
      for (int w = 0; w < input_width; w++) {
        for (int c = 0; c < input_chan; c++) {
          //NHWC (needs RBswap)
          const size_t offset_cv =
            h * strides_cv[1] + w * strides_cv[2] + (input_chan - c - 1) * strides_cv[3];
          //NCHW
          const size_t offset = n * strides[0] + (c)*strides[1] + h * strides[2] + w * strides[3];
          input_h_[offset] =
            (static_cast<float>(dst_images[n].data[offset_cv]) / 255.0f - mean[c]) / std[c];
        }
      }
    }
  }
  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    input_d_.get(), input_h_.data(), input_h_.size() * sizeof(float), cudaMemcpyHostToDevice,
    *stream_));
}

bool TrtDepth_Anything::doInference(const std::vector<cv::Mat> & images)
{
  if (!trt_common_->isInitialized()) {
    return false;
  }
  preprocess(images);
  //return infer(images);
  return infer();
}

//bool TrtDepth_Anything::infer(const std::vector<cv::Mat> & images)
bool TrtDepth_Anything::infer()
{
  std::vector<void *> buffers = {input_d_.get(), out_prob_d_.get()};
  trt_common_->enqueueV2(buffers.data(), *stream_, nullptr);

  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    out_prob_h_.get(), out_prob_d_.get(), sizeof(float) * out_elem_num_, cudaMemcpyDeviceToHost,
    *stream_));
  cudaStreamSynchronize(*stream_);
  return true;
}

cv::Mat TrtDepth_Anything::getDepthImage()
{
  const auto output_dims = trt_common_->getBindingDimensions(1);
  int width = output_dims.d[3];
  int height = output_dims.d[2];
  //float * ptr = out_prob_h_.get();
  cv::Mat depthImage(height, width, CV_32FC1, out_prob_h_.get());
  cv::normalize(depthImage, depthImage, 0, 255, cv::NORM_MINMAX, CV_8U);
  return depthImage;
}
}  // namespace tensorrt_depth_anything
