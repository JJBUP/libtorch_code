#include <torch/torch.h>
#include "linear_regression.h"

LinearRegression::LinearRegression(int64_t input_size, int64_t output_size)
{
    linear_layer = torch::nn::Sequential(
        torch::nn::Linear(input_size, output_size),
        torch::nn::BatchNorm1d(output_size),
        torch::nn::ReLU());
    classifier = torch::nn::Linear(output_size, output_size);
    register_module("linear_layer", linear_layer);
    register_module("classifier", classifier);
}
torch::Tensor LinearRegression::forward(torch::Tensor x)
{
    x = linear_layer->forward(x);
    x = classifier(x);
    return x;
}

// class LinearRegression : public torch::nn::Module
// {
// private:
//     torch::nn::Sequential linear_layer;
//     torch::nn::Linear classfier;

// public:
//     // LinearRegression(int64_t input_size, int64_t output_size) : // 初始化列表
//     //     linear(register_module("linear", torch::nn::Linear(input_size, output_size))) {}
//     LinearRegression(int64_t input_size, int64_t output_size)
//     {
//         this->linear_layer = torch::nn::Sequential(

//             torch::nn::Linear(input_size, output_size),
//             torch::nn::BatchNorm1d(output_size),
//             torch::nn::ReLU());
//         this->classfier = torch::nn::Linear(output_size, output_size);
//         register_module("linear_layer", linear_layer);
//         register_module("classfier", classfier);
//     }

//     torch::Tensor forward(torch::Tensor x)
//     {
//         x = linear_layer->forward(x);
//         x = classfier(x);
//         return x;
//     }
// };
