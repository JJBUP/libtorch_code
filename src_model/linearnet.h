#include <torch/torch.h>
class LinearRegression : public torch::nn::Module
{
private:
    torch::nn::Sequential linear_layer = nullptr;
    torch::nn::Linear classifier = nullptr;

public:
    LinearRegression(int64_t input_size, int64_t output_size);
    torch::Tensor forward(torch::Tensor x);
};