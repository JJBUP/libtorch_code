#include <torch/torch.h>
#include "linearnet.h"

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
