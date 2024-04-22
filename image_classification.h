#include <torch/torch.h>
class ImageClassification : public torch::nn::Module
{
    // int in_channel;
    // int class_num;
    // int stage;
    // ModuleList不好用,其默认存储为Torch::nn::Module类，但是该类没有forward方法
    // torch::nn::ModuleList conv_list = nullptr; 
    // torch::nn::ModuleList fc_list = nullptr;
private:
    torch::nn::Sequential convs = nullptr;
    torch::nn::Sequential fcs = nullptr;

public:
    ImageClassification(int in_channel, int class_num, int stage);
    torch::Tensor forward(torch::Tensor x);
};