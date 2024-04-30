#include "image_dataset.h"
#include <torch/torch.h>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <string>
// 阿尔兹海默症分类
ImageDataset::ImageDataset(const std::string &root_dir, const std::map<std::string, int> &class_name, std::string mode = "train") //,const std::vector<std::string> &transforms)
{
  cls = class_name;
  // 获取目录地址
  rd = root_dir;
  md = mode;
  if (md == "train")
  {
    recursive_rglob(rd.append("/train"), &paths);
  }
  else
  {
    recursive_rglob(rd.append("/test"), &paths);
  }

  long height = 0;
  long width = 0;
  for (auto &path : paths)
  {
    cv::Mat img = cv::imread(path);
    height = height + img.rows;
    width = width + img.cols;
  }
  average_height = height / paths.size();
  average_width = width / paths.size();
}

torch::data::Example<torch::Tensor, torch::Tensor> ImageDataset::get(size_t index)
{
  // 如果像传递多个值，则需要使用std::tuple<T1, T2, ...>或重写结构体模板
  // torch::data::Example 具有默认类型值的结构体模板
  std::filesystem::path path = paths[index];
  // 获取类别名称
  std::string label_string = path.parent_path().stem().string();
  long label = cls[label_string];
  // 读取灰度图片
  cv::Mat img = cv::imread(path.string(), cv::IMREAD_GRAYSCALE);
  // cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
  // bug注意：
  // 1. 同一批次的图片大小相同，否则dataloader将返回vector包装的张量
  // 2. resize可能会导致转换后的tensor出现nan值
  // cv::resize 使用插值的方式改变图片大小 而img.resize 则不是使用插值的方式改变
  cv::resize(img, img, cv::Size(average_width, average_height), cv::INTER_LINEAR);
  img.convertTo(img, CV_32FC1, 1.0f / 255.0f);
  torch::Tensor img_ = torch::from_blob(img.data, {1, average_width, average_height}, torch::kFloat32).clone();
  torch::Tensor label_ = torch::tensor(label, torch::kLong);
  // 检查张量中是否存在NaN值

  return {img_, label_};
}
// torch::optional<> 用于表示可选值,即值可能存在或不存在
std::optional<size_t> ImageDataset::size() const
{ // 注意const修饰符也要一致，函数修饰符
  return paths.size();
}
// 递归遍历目录
void ImageDataset::recursive_rglob(const std::string &directory_path, std::vector<std::string> *const paths_ptr)
{
  for (const auto &entry : std::filesystem::directory_iterator(directory_path))
  {
    if (std::filesystem::is_regular_file(entry))
    {
      // std::cout << entry.path() << std::endl; // 输出文件的路径
      paths_ptr->push_back(entry.path());
    }
    else if (std::filesystem::is_directory(entry))
    {
      recursive_rglob(entry.path(), paths_ptr); // 递归遍历子目录
    }
  }
}
