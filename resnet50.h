#include <torch/torch.h>

class Resnet50 : public torch::nn::Module
{
private:
    /* data */
    std::vector<torch::nn::Sequential> seq_list;
    torch::nn::Sequential seq_first = nullptr;
    torch::nn::Sequential seq_last = nullptr;

public:
    Resnet50(const int in_channel, const int num_classes);
    torch::Tensor forward(torch::Tensor x);
};