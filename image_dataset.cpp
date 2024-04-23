#include "image_dataset.h"
#include <torch/torch.h>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <string>
// 阿尔兹海默症分类
ImageDataset::ImageDataset(const std::string &root_dir, const std::map<std::string, int> &class_name, const std::vector<std::string> &transforms)
{
  cls = class_name;
  tfs = transforms;
  // 获取目录地址
  rd = root_dir;
  for (const auto &entry : std::filesystem::directory_iterator(root_dir))
  {
    if (!entry.is_directory())
    {
      paths.push_back(entry.path());
    }
  }
}
torch::data::Example<> ImageDataset::get(size_t index)
{
  // 如果像传递多个值，则需要使用std::tuple<T1, T2, ...>或重写结构体模板
  // torch::data::Example 具有默认类型值的结构体模板
  std::filesystem::path path = paths[index];
  // 获取类别名称
  std::string label_staing = path.parent_path().stem().string();
  long label = cls[label_staing];
  // 读取图片
  cv::Mat img = cv::imread(path.string());
  cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
  // cv::resize(img, img, cv::Size(224, 224));
  img.convertTo(img, CV_32FC3, 1.0 / 255.0);
  torch::Tensor img_ = torch::from_blob(img.data, {img.rows, img.cols, 3}, torch::kFloat32);
  torch::Tensor label_ = torch::tensor(label, torch::kLong);
  return {img_, label_};
}