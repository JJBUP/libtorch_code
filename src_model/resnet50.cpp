    #include <torch/torch.h>
    #include "resnet50.h"
    #include <iostream>

    torch::nn::Sequential make_conv_layer(int in_channel, int out_channel, int ks, int deep_n)
    {
        torch::nn::Sequential seq = torch::nn::Sequential();
        int in_c = in_channel;
        for (int i = 0; i < deep_n; i++)
        {
            if (i == 0)
            {
                in_c = in_channel;
            }
            else
            {
                in_c = out_channel;
            }

            seq->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(in_c, in_channel, 1).padding(0)));
            seq->push_back(torch::nn::BatchNorm2d(in_channel));
            seq->push_back(torch::nn::ReLU());
            seq->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channel, in_channel, ks).padding(1)));
            seq->push_back(torch::nn::BatchNorm2d(in_channel));
            seq->push_back(torch::nn::ReLU());
            seq->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channel, out_channel, 1).padding(0)));
            seq->push_back(torch::nn::ReLU());
        }
        return seq;
    }

    Resnet50::Resnet50(const int in_channel, const int num_classes)
    {
        std::vector<int> kernel_size = {3, 3, 3, 3};
        std::vector<int> deep_num = {3, 4, 6, 3};
        std::vector<int> out_channel = {128, 256, 512, 1024};
        seq_first = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channel, 64, 3).padding(1)),
            torch::nn::BatchNorm2d(64),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).padding(1).stride(2)));
        seq_list.push_back(make_conv_layer(64, out_channel[0], kernel_size[0], deep_num[0]));
        seq_list.push_back(make_conv_layer(out_channel[0], out_channel[1], kernel_size[1], deep_num[1]));
        seq_list.push_back(make_conv_layer(out_channel[1], out_channel[2], kernel_size[2], deep_num[2]));
        seq_list.push_back(make_conv_layer(out_channel[2], out_channel[3], kernel_size[3], deep_num[3]));
        seq_last = torch::nn::Sequential(
            torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions(1)),
            torch::nn::Flatten(),
            torch::nn::Linear(out_channel[3], num_classes));
        register_module("seq_first", seq_first);
        register_module("seq_last", seq_last);
        int i = 0;
        for (torch::nn::Sequential &seq : seq_list)
        {
            register_module("seq_list_" + std::to_string(i), seq);
            i++;
        }
    }
    torch::Tensor Resnet50::forward(torch::Tensor x)
    {
        x = seq_first->forward(x);
        for (torch::nn::Sequential seq : seq_list)
        {
            x = seq->forward(x);
        }
        return seq_last->forward(x);
    }
