#include <torch/torch.h>
#include "image_classification.h"

ImageClassification::ImageClassification(int in_channel, int class_num, int stage = 4)
{   
    // 忘记初始化
    convs = torch::nn::Sequential();
    int in_ch = in_channel;
    for (int i = 0; i < stage; i++)
    {
        int out_ch = 8 * std::pow(2, stage); // 8 16 32 64
        convs->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(in_ch, out_ch, 3).padding(1)));
        convs->push_back(torch::nn::BatchNorm2d(out_ch));
        convs->push_back(torch::nn::ReLU());
        convs->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));
        in_ch = out_ch;
    }
    fcs = torch::nn::Sequential(); //忘记初始化
    fcs->push_back(torch::nn::Linear(torch::nn::LinearOptions(in_ch, int(in_ch / 2))));
    fcs->push_back(torch::nn::BatchNorm1d(int(in_ch / 2)));
    fcs->push_back(torch::nn::ReLU());

    fcs->push_back(torch::nn::Linear(torch::nn::LinearOptions(int(in_ch / 2), int(in_ch / 4))));
    fcs->push_back(torch::nn::BatchNorm1d(int(in_ch / 4)));
    fcs->push_back(torch::nn::ReLU());
    fcs->push_back(torch::nn::Linear(torch::nn::LinearOptions(int(in_ch / 4), class_num)));
    // 注册模块
    register_module("convs", convs);
    register_module("fcs", fcs);
}
torch::Tensor ImageClassification::forward(torch::Tensor x)
{
    x = convs->forward(x); // bchw
    x = torch::mean(torch::mean(x, -1), -1); // bc
    x = fcs->forward(x); // bn_class
    return x;
}