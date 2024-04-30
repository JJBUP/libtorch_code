#include <torch/torch.h>
// 查看源码可知：<>的作用是为了将ImageDataset传递给Dataset模板类以便
class Lenet5 : public torch::nn::Module
{
public:
    Lenet5();
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Sequential seq = nullptr;
    torch::nn::Conv2d conv1 = nullptr, conv2 = nullptr;
    torch::nn::ReLU relu = nullptr;
    torch::nn::MaxPool2d max_pool = nullptr;
    torch::nn::Linear lin1 = nullptr, lin2 = nullptr, lin3 = nullptr;
};
