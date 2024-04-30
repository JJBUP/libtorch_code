#include "lenet.h"
#include <torch/torch.h>
// 构建函数
Lenet5::Lenet5()
{
  conv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 6, 5).padding(2));
  conv2 = torch::nn::Conv2d(torch::nn::Conv2dOptions(6, 16, 5));
  relu = torch::nn::ReLU();
  max_pool = torch::nn::MaxPool2d(2);
  lin1 = torch::nn::Linear(5 * 5 * 16, 120);
  lin2 = torch::nn::Linear(120, 84);
  lin3 = torch::nn::Linear(84, 10);
}

// 前向传播
torch::Tensor Lenet5::forward(torch::Tensor x) 
{
  x = max_pool(relu(conv1(x)));
  x = max_pool(relu(conv2(x)));
  x = x.reshape({x.size(0), -1});
  x = lin1(x);
  x = lin2(x);
  x = lin3(x);
  return x;
}