#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "linearnet.h"
#include "imgclsnet.h"
#include "image_dataset.h"
#include "resnet50.h"
void tensor_type();
void tensor_create();
void tensor_dim();
void tensor_index();
void tensor_operation();
void auto_grad();
void linear_regression();
void img_classification();
void alzheimer_s_classification(int batch_size);
void jit_script_test();
void object_load_and_save();
float accuracy_compute(torch::Tensor pl, torch::Tensor l);

int main(int argc, char *argv[])
{
    // tensor_type();
    // tensor_dim();
    // tensor_create();
    // auto_grad();
    // tensor_index();
    // tensor_operation();
    // linear_regression();
    img_classification();
    // test_dataset();
    // jit_script_test();
    // object_load_and_save();
    // if (argc == 1)
    // {
    //     std::cout << "argc = " << argc << std::endl;
    //     alzheimer_s_classification(2);
    // }
    // else if (argc == 2)
    // {
    //     std::cout << "argc = " << argc << std::endl;
    //     std::cout << "argv[1] = " << std::stoi(argv[1]) << std::endl;
    //     alzheimer_s_classification(std::stoi(argv[1]));
    // }
    // jit_script_test();
}

void tensor_type()
{
    torch::Tensor a = torch::arange(12);
    a = a.reshape({2, 2, -1});
    // 查看数据类型
    std::cout << "a.dtype() = " << a.dtype() << std::endl;
    // 修改数据类型
    torch::Tensor a1 = a.to(torch::kFloat32);
    torch::Tensor a2 = a.toType(torch::kFloat32);
    torch::Tensor a3 = a.type_as(a1);
    std::cout << "a1.to() = " << a1.dtype() << std::endl;
    std::cout << "a2.toType() = " << a2.dtype() << std::endl;
    std::cout << "a3.type_as() = " << a3.dtype() << std::endl;
}

void tensor_dim()
{

    torch::Tensor a = torch::arange(12).reshape({2, 2, -1});
    std::cout << "a.dim() = " << a.dim() << std::endl; // dim() 函数,获取维度数量,int64_t对象
    std::cout << a.view({3, 2}) << std::endl;          // view() 函数,获取重塑张量
    std::cout << a.reshape({3, 2}) << std::endl;       // reshape() 函数,获取重塑张量
    std::cout << a.flatten() << std::endl;             // flatten() 函数,获取展平张量
    std::cout << a.unsqueeze(0) << std::endl;          // unsqueeze() 函数,获取增加维度张量
    std::cout << a.squeeze(0) << std::endl;            // squeeze() 函数,获取删除维度张量
    std::cout << a.transpose(0, 1) << std::endl;       // transpose() 函数,获取转置张量
    std::cout << a.permute({1, 0}) << std::endl;       // permute() 函数,获取转置张量
}
// 创建tensor
void tensor_create()
{
    // 基础数据创建
    torch::Tensor a = torch::zeros({2, 3}, torch::TensorOptions().dtype(torch::kFloat32));
    torch::Tensor b = torch::ones({2, 3}, torch::TensorOptions().device(torch::kCPU));
    torch::Tensor c = torch::eye({3});
    torch::Tensor d = torch::full({2, 3}, 3.14);
    torch::Tensor e = torch::rand({2, 3});  // uniform distribution, 0-1的均匀分布
    torch::Tensor f = torch::randn({2, 3}); // normal distribution
    torch::Tensor g = torch::arange(10, 20, 2);
    torch::Tensor h = torch::tensor({{1, 2}, {3, 4}});
    std::cout << a << std::endl;
    std::cout << b << std::endl;
    std::cout << c << std::endl;
    std::cout << d << std::endl;
    std::cout << e << std::endl;
    std::cout << f << std::endl;
    std::cout << g << std::endl;
    std::cout << h << std::endl;
    // 配置张量参数
}
// Tensor Indexing API : torch::Tensor::index,增删改查操作

void tensor_index()
{

    torch::Tensor a = torch::arange(12);
    a = a.reshape({2, 2, -1});

    // 广播机制
    std::cout << "广播机制测试：" << std::endl
              << a + torch::tensor({200, 200, 200}) << std::endl;
    // 1.查 Getter(获取器):C++ 使用 Tensor中成员函数index来实现python的“[]”切片
    torch::Tensor b = a.index({0});                                              // 对应python中的[0]
    torch::Tensor c = a.index({0, 1});                                           // 对应python中的[0,1]
    torch::Tensor d = a.index({0, 1, -1});                                       // 对应python中的[0,1,-1]
    torch::Tensor e = a.index({torch::indexing::Slice(), 1, 2});                 // Slice(int start,int end)类用于设定单一维度切片范围,Slice()对应python中的:
    torch::Tensor f = a.index({torch::indexing::Slice(0, -1), 1, 2});            // Slice(int start,int end)类用于设定单一维度切片范围,Slice()对应python中的:
    torch::Tensor g = a.index({"...", 2});                                       // "..."对应python中的...
    torch::Tensor h = a.index({torch::indexing::Ellipsis, 2});                   // Ellipsis对应python中的...
    torch::Tensor i = a.index({torch::indexing::None});                          // None对应python中的None,增加一个新的维度
    torch::Tensor j = torch::rand({2, 3}).to(torch::kBool).index({"...", true}); // kBool为ScalarType基础张量类型,true为python中的True
    torch::Tensor k = a.index({torch::tensor({0, 0, 0}),
                               torch::tensor({1, 1, 1}), torch::tensor({2, 2, 2})}); // int类型tensor作为索引
    std::cout << "a_index:" << std::endl
              << a << std::endl;
    std::cout << "b_index:" << std::endl
              << b << std::endl;
    std::cout << "c_index:" << std::endl
              << c << std::endl;
    std::cout << "d_index:" << std::endl
              << d << std::endl;
    std::cout << "e_index:" << std::endl
              << e << std::endl;
    std::cout << "f_index:" << std::endl
              << f << std::endl;
    std::cout << "g_index:" << std::endl
              << g << std::endl;
    std::cout << "h_index:" << std::endl
              << h << std::endl;
    std::cout << "i_index.size():" << std::endl
              << i.sizes() << std::endl;
    std::cout << "j_index:" << std::endl
              << j << std::endl;
    std::cout << "k_index:" << std::endl
              << k << std::endl;

    std::cout << std::endl
              << "<<<<<setter<<<<<<" << std::endl;

    // 2. 改  Setter(设置器)：
    // 重置索引位置值
    // index_put_原变量修改,index_put返回新的变量
    std::cout << "b:" << std::endl
              << a.clone().index_put_({0}, -1) << std::endl;
    std::cout << "c:" << std::endl
              << a.clone().index_put_({0, 1}, -1) << std::endl;
    std::cout << "d:" << std::endl
              << a.clone().index_put_({0, 1, -1}, -1) << std::endl;
    std::cout << "e:" << std::endl
              << a.clone().index_put_({torch::indexing::Slice(), 1, 2}, -1) << std::endl;
    std::cout << "f:" << std::endl
              << a.clone().index_put_({torch::indexing::Slice(0, -1), 1, 2}, -1) << std::endl;
    std::cout << "g:" << std::endl
              << a.clone().index_put_({"...", 2}, -1) << std::endl;
    std::cout << "h:" << std::endl
              << a.clone().index_put_({torch::indexing::None}, -1) << std::endl; // None对应python中的None,增加一个新的维度,即全部设置为-1
    std::cout << "j:" << std::endl
              << a.clone().index_put_({torch::indexing::Ellipsis, 2}, -1) << std::endl;
    std::cout << "k:" << std::endl
              << a.clone().index_put_({torch::tensor({0, 0, 0}), torch::tensor({1, 1, 1}), torch::tensor({2, 2, 2})}, -1) << std::endl;

    // 索引位置，两个张量规约（prod, mean, amax or amin）
    // 指定索引相加,注意待加张量与索引后的张量的维度必须一致
    torch::Tensor indices = torch::tensor({0});
    torch::Tensor values = torch::full({indices.size(0), a.size(1), a.size(2)}, 100);
    std::cout << "b_reduce:" << std::endl
              << a.clone().index_reduce_(0, indices, values, "amax");
    c = a.index_reduce(0, indices, values, "amax");
    std::cout << "c_reduce:" << std::endl
              << c << std::endl;
    // 索引位置,两个张量相加（比较鸡肋，因为简单的加减操作index_put_完全可以做到）
    // 指定索引相加,注意待加张量与索引后的张量的维度必须一致
    indices = torch::tensor({0, 1});
    values = torch::full({a.size(0), a.size(1), indices.size(0)}, 100);
    std::cout << "b:" << std::endl
              << a.clone().index_add_(-1, indices, values); // 修改原变量
    c = a.index_add(-1, indices, values);                   // 不修改原变量
    std::cout << "c:" << std::endl
              << c << std::endl;

    // 其他的方法就比较鸡肋了,index_reduce,index_put和index可完成其他index开头方法的操作
}

void tensor_operation()
{
    // 获取属性
    // 除了 shape 属性（或者 size() 函数）变为了 sizes() 函数，其余常用的基本上和 PyTorch 一致
    torch::Tensor a = torch::rand({2, 3});
    std::cout << a.size(0) << a.size(1) << std::endl; // size() 函数,获取单一维度大小
    std::cout << a.sizes() << std::endl;              // sizes() 函数,获取维度大小,array对象
    std::cout << a.dim() << std::endl;                // dim() 函数,获取维度数量,int64_t对象
    std::cout << a.device() << std::endl;             // device() 函数,获取设备类型
    std::cout << a.dtype() << std::endl;              // dtype() 函数,获取数据类型
    std::cout << a.requires_grad() << std::endl;      // requires_grad() 函数,获取是否需要梯度
    std::cout << a.data() << std::endl;               // data() 函数,获取数据指针
    std::cout << a.grad() << std::endl;               // grad() 函数,获取梯度
    // 设备转换
    std::cout << a.to(torch::kCUDA) << std::endl;                // to() 函数,获取指定设备类型张量
    std::cout << a.to(torch::Device(torch::kCUDA)) << std::endl; // to() 函数,获取指定设备类型张量
    std::cout << a.to(torch::Device("cuda:0")) << std::endl;     // to() 函数,获取指定设备类型张量
    std::cout << a.to(torch::kCPU) << std::endl;                 // to() 函数,获取指定设备类型张量
    std::cout << a.cuda() << std::endl;                          // cuda() 函数,获取cuda张量
    std::cout << a.cpu() << std::endl;                           // cuda() 函数,获取cuda张量

    std::cout << a.detach() << std::endl;                 // detach() 函数,获取不带梯度张量
    std::cout << a.clone() << std::endl;                  // clone() 函数,获取克隆张量
    std::cout << a[0][0].item<float>() << std::endl;      // item() 函数,获取数据
    std::cout << a[0][0].item() << std::endl;             // item() 函数,获取数据
    std::cout << a.cumsum(0) << std::endl;                // cumsum() 函数,获取累加张量
    std::cout << a.cumprod(0) << std::endl;               // cumprod() 函数,获取累乘张量
    std::cout << std::get<0>(a.topk(1, -1)) << std::endl; // topk() 函数,获取topk张量数值
    std::cout << std::get<1>(a.topk(1, -1)) << std::endl; // topk() 函数,获取topk张量索引
    std::cout << a.contiguous() << std::endl;             // contiguous() 函数,获取连续张量
    std::cout << torch::cat({a, a}, 1) << std::endl;      // cat() 函数,获取拼接张量
    std::cout << torch::stack({a, a}, 1) << std::endl;    // stack() 函数,获取堆叠张量
    // std::cout << a.to_sparse() << std::endl;                     // to_sparse() 函数,获取稀疏张量
    // std::cout << a.to_dense() << std::endl;                      // to_dense() 函数,获取稠密张量
    // std::cout << a.to_sparse_csr() << std::endl;                 // to_sparse_csr() 函数,获取稀疏张量
    // std::cout << a.to_sparse_csc() << std::endl;                 // to_sparse_csc() 函数,获取稀疏张量

    // 数学计算
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> svd_data = a.svd();
    std::cout << std::get<0>(svd_data) << std::get<1>(svd_data) << std::get<2>(svd_data) << std::endl; // svd() 函数,获取SVD张量
    std::cout << a.t() << std::endl;
    std::cout << a.argmax() << std::endl;
    std::tuple<torch::Tensor, torch::Tensor> max_idx_val = a.max(1, true);
    std::cout << std::get<0>(max_idx_val) << std::get<1>(max_idx_val) << std::endl;
    std::cout << a.amax() << std::endl;
    std::cout << a.where(a > 0.5, -1) << std::endl;
    std::cout << a.clamp(0, 1) << std::endl; // clamp() 函数,获取限制在指定范围内的张量
    std::cout << a.ceil() << std::endl;      // ceil() 函数,获取向上取整张量
    std::cout << a.floor() << std::endl;     // floor() 函数,获取向下取整张量
    std::cout << a.round() << std::endl;     // round() 函数,获取四舍五入张量
    std::cout << a.fmod(0.5) << std::endl;   // fmod() 函数,获取取模张量
    std::cout << a.div(0.5) << std::endl;    // div() 函数,获取除法张量

    // 判断
    std::cout << a.is_cuda() << std::endl;           // is_cuda() 函数,获取是否在GPU上
    std::cout << a.is_cpu() << std::endl;            // is_cpu() 函数,获取是否在CPU上
    std::cout << a.is_floating_point() << std::endl; // is_floating_point() 函数,获取是否是浮点数
    std::cout << a.is_same(a) << std::endl;          // is_same() 函数,获取是否是相同的张量
    std::cout << a.isnan() << std::endl;             // isnan() 函数,获取是否是NaN
    std::cout << a.isinf() << std::endl;             // isinf() 函数,获取是否是无穷大
    std::cout << a.isfinite() << std::endl;          // isfinite() 函数,获取是否是有穷
    std::cout << a.all() << std::endl;               // all() 函数,获取是否全部为真
    std::cout << a.any() << std::endl;               // any() 函数,获取是否任意为真
    std::cout << a.equal(a) << std::endl;            // equal() 函数,获取是否相等
    std::cout << a.is_nonzero() << std::endl;        // is_nonzero() 函数,获取是否非零
    std::cout << a.is_same_size(a) << std::endl;     // is_same_size() 函数,获取是否相同大小
}

// 自动微分
void auto_grad()
{
    torch::Tensor a = torch::full({3, 3}, 2, torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false)).to(torch::Device("cuda:0"));
    torch::Tensor b = torch::full({3, 3}, 3, torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false)).to(torch::kCUDA);
    torch::Tensor c = torch::full({3, 3}, 4, torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false)).cuda();
    std::cout << "a.grad: " << a.grad() << std::endl;
    // 返回是否可导
    std::cout << "a.req_grad:" << a.requires_grad() << std::endl;
    std::cout << "b.req_grad:" << b.requires_grad() << std::endl;
    // 修改是否可导
    // a.requires_grad_(true);
    b.requires_grad_(true);
    c.requires_grad_(true);
    torch::Tensor d = c * (a * b);
    torch::Tensor e = d.sum();
    d.retain_grad(); // 保留梯度
    e.retain_grad(); // 保留梯度
    e.backward();    // 反向传播
    std::cout << "a.grad:" << a.grad() << std::endl;
    std::cout << "b.grad:" << b.grad() << std::endl;
    std::cout << "c.grad:" << c.grad() << std::endl;
    std::cout << "d.grad:" << d.grad() << std::endl; // 非叶子节点
    std::cout << "e.gard:" << e.grad() << std::endl; // 非叶子节点
    // e.backward();    // 第二次反向传播会因计算图消失而报错,需要设置retain_graph(true)
}

// 模拟线性回归
void linear_regression()
{
    torch::manual_seed(1);
    int input_size = 5;
    int output_size = 10;

    std::shared_ptr<LinearRegression> model_ptr = std::make_shared<LinearRegression>(LinearRegression(input_size, output_size));
    torch::optim::SGD optimizer(model_ptr->parameters(), torch::optim::SGDOptions(0.01).momentum(0.9));
    for (int epoch = 0; epoch < 100; epoch++)
    {
        torch::Tensor a = torch::randn({1, 5}, torch::requires_grad(true));
        torch::Tensor x = torch::randn({10, 5}, torch::requires_grad(true)) * 100 + 100;
        torch::Tensor b = torch::randn({10, 1}, torch::requires_grad(true));
        torch::Tensor y = torch::matmul(x, a.t()) + b; // 10 * 1
        // std::cout << "x: " << std::endl
        //           << x.sizes() << std::endl
        //           << x << std::endl;
        // std::cout << "y: " << std::endl
        //           << y.sizes() << std::endl
        //           << y << std::endl;
        optimizer.zero_grad();
        torch::Tensor y_pred = model_ptr->forward(x);
        torch::Tensor loss = torch::mse_loss(y_pred, y);
        loss.backward();
        optimizer.step();
        std::cout << "loss: " << loss.item<float>() << std::endl;
        std::cout << "epoch: " << epoch << " loss: " << loss.item<float>() << std::endl;
    }
}

// 模拟图像分类
void img_classification()
{
    int image_size[2] = {64, 64};
    int rgb_channel = 3;

    std::shared_ptr<ImageClassification> model_ptr = std::make_shared<ImageClassification>(ImageClassification(rgb_channel, 5, 4));
    torch::optim::SGD optimizer(model_ptr->parameters(), torch::optim::SGDOptions(0.001).momentum(0.9));
    for (int epoch = 0; epoch < 100; epoch++)
    {
        // 输入张量
        cv::Mat img_ = cv::imread("/home/jjb/libtorch_code/data/test.jpg", cv::IMREAD_COLOR);
        cv::cvtColor(img_, img_, cv::COLOR_BGR2RGB);
        img_.resize(image_size[0], image_size[1]);
        img_.convertTo(img_, CV_32FC3, 1.0f / 255.0f); // 转换为RGB格式,注意
        // 注意cv数据类型要与from_blob获得Tnesor数据类型一致,内存数据才能正确转换
        torch::Tensor img = torch::from_blob(img_.data, {1, rgb_channel, image_size[0], image_size[1]}, torch::TensorOptions().dtype(torch::kFloat32));
        // 模拟批量
        img = img.repeat({10, 1, 1, 1});
        torch::Tensor label = torch::one_hot(torch::ones(10, torch::kInt64), 5).toType(torch::kFloat32);
        // std::cout << "Image Size: " << std::endl
        //           << img.sizes() << std::endl;
        torch::Tensor pred = model_ptr->forward(img);
        // std::cout << "pred:" << std::endl
        //           << pred << std::endl
        //           << pred.sizes();
        torch::Tensor loss = torch::cross_entropy_loss(pred, label);
        loss.backward();
        optimizer.step();
        std::cout << "epoch: " << epoch << " loss: " << loss.item<float>() << std::endl;
    }

    // test
    cv::Mat img_ = cv::imread("/home/jjb/libtorch_code/data/test.jpg", cv::IMREAD_COLOR);
    cv::cvtColor(img_, img_, cv::COLOR_BGR2RGB);
    img_.resize(image_size[0], image_size[1]);
    img_.convertTo(img_, CV_32FC3, 1.0f / 255.0f);
    torch::Tensor img = torch::from_blob(img_.data, {1, rgb_channel, image_size[0], image_size[1]}, torch::TensorOptions().dtype(torch::kFloat32));

    img = img.repeat({10, 1, 1, 1});
    torch::Tensor pred = model_ptr->forward(img);
    torch::Tensor result = pred.softmax(1).amax(1);
    std::cout << "result:" << std::endl
              << result << std::endl;
}

void test_dataset()
{
    // 元数据
    std::string root_dir = "../data/Alzheimer_s Dataset";
    std::map<std::string, int> class_id = {{"MildDemented", 0},
                                           {"ModerateDemented", 1},
                                           {"NonDemented", 2},
                                           {"VeryMildDemented", 3}};
    // ImageDataset *dataset_ptr = new ImageDataset(root_dir, class_id, "train");
    std::shared_ptr<ImageDataset> dataset_ptr = std::make_shared<ImageDataset>(ImageDataset(root_dir, class_id, "train"));

    torch::data::Example<> dl = dataset_ptr->get(0);
    std::optional<size_t> data_len = dataset_ptr->size(); // 总结一下optioal容器的用法
    std::cout << "dataset_ptr->size(): " << data_len.value() << std::endl
              << "data.data.sizes(): " << dl.data.sizes() << std::endl
              << "data.target: " << dl.target << std::endl;
}

void object_load_and_save()
{
    std::shared_ptr<Resnet50> resnet50_ptr = std::make_shared<Resnet50>(Resnet50(3, 3));
    torch::serialize::OutputArchive archive_out; // 创建输出archive
    resnet50_ptr->save(archive_out);             // 将模型参数保存到archive
    archive_out.save_to("../logs/resnet.pt");    // 将archive保存到文件
    std::cout << "save model success" << std::endl;
    torch::serialize::InputArchive archive_in;                                                // 创建输入archive
    archive_in.load_from("../logs/resnet.pt");                                                // 从文件加载archive
    std::shared_ptr<Resnet50> resnet50_ptr_load = std::make_shared<Resnet50>(Resnet50(3, 3)); // 创建一个新的模型
    resnet50_ptr_load->load(archive_in);                                                      // 从archive加载模型参数
    std::cout << "load model success" << std::endl;
}

void alzheimer_s_classification(int batch_size = 2)
{

    // 元数据
    int epoch = 100;
    torch::Device device = torch::Device("cuda:1");
    std::string root_dir = "../data/alzheimer_dataset";
    std::map<std::string, int> class_id = {{"MildDemented", 0},
                                           {"ModerateDemented", 1},
                                           {"NonDemented", 2},
                                           {"VeryMildDemented", 3}};
    // 数据读取
    std::shared_ptr<ImageDataset> train_dataset_ptr = std::make_shared<ImageDataset>(ImageDataset(root_dir, class_id, "train"));
    std::shared_ptr<ImageDataset> val_dataset_ptr = std::make_shared<ImageDataset>(ImageDataset(root_dir, class_id, "test"));
    std::cout << "训练数据量: " << train_dataset_ptr->size().value() << std::endl;
    std::cout << "验证数据量: " << val_dataset_ptr->size().value() << std::endl;
    // 数据加载
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(*train_dataset_ptr, torch::data::DataLoaderOptions().workers(4).batch_size(batch_size));
    auto val_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(*val_dataset_ptr, torch::data::DataLoaderOptions().workers(4).batch_size(batch_size).enforce_ordering(true));
    // 计算批次数量
    int train_batch_count = 0;
    for (const auto &batch : *train_loader)
    {
        train_batch_count++;
    }
    // 计算批次数量
    int val_batch_count = 0;
    for (const auto &batch : *val_loader)
    {
        val_batch_count++;
    }

    // // 模型
    std::shared_ptr<Resnet50> resnet50_ptr = std::make_shared<Resnet50>(Resnet50(1, 4));
    resnet50_ptr->to(device);
    // // 优化器
    torch::optim::SGD optimizer(resnet50_ptr->parameters(), torch::optim::SGDOptions(0.05).momentum(0.9));
    // 学习率调整,目前只提供了StepLR,实现其他的学习率调整方式可以参考torch::optim::StepLR
    torch::optim::StepLR lr_scheduler(optimizer, 30, 0.1); // 学习率调整, 30轮后学习率乘以0.1
    float val_acc = 0;
    for (int i = 0; i < epoch; i++)
    {
        std::cout << ">>>>>>> train stage - epoch: " << i << "<<<<<<<" << std::endl;
        resnet50_ptr->train(); // 设置为训练模式
        float train_loss = 0;
        int train_correct = 0;
        int tain_batch_num = 0;
        for (const std::vector<torch::data::Example<>> &batch : *train_loader)
        { // 返回vector包装的批次数据
            std::vector<torch::Tensor> imgs;
            std::vector<torch::Tensor> labels;
            for (torch::data::Example<> data_label : batch)
            {
                imgs.push_back(data_label.data);
                labels.push_back(data_label.target);
            }
            torch::Tensor imgs_t = torch::stack(torch::TensorList(imgs)).to(device);
            torch::Tensor labels_t = torch::stack(torch::TensorList(labels)).to(device);
            // 训练
            torch::Tensor pred = resnet50_ptr->forward(imgs_t);

            optimizer.zero_grad();
            torch::Tensor loss = torch::cross_entropy_loss(pred, labels_t);
            loss.backward();
            optimizer.step();
            // 后处理
            torch::Tensor pred_label = torch::argmax(pred, 1);
            train_loss += loss.item<float>();
            train_correct = train_correct + torch::sum(pred_label == labels_t).item<float>();
            tain_batch_num++;
            std::cout << " epoch: " << i
                      << " batch_num: " << tain_batch_num << " / " << train_batch_count
                      << " batch_size: " << batch.size()
                      << " train_acc: " << torch::sum(pred_label == labels_t).item<float>() / labels_t.size(0)
                      << " batch_loss: " << loss.item<float>() << std::endl;
        }
        lr_scheduler.step();
        std::cout << ">>>>>>> train stage - result - epoch: " << i << "<<<<<<<" << std::endl;
        std::cout << " epoch: " << i
                  << " train accuracy: " << train_correct / float(train_dataset_ptr->size().value())
                  << " batch_loss: " << train_loss / tain_batch_num << std::endl;
        // 验证
        std::cout << ">>>>>>> val stage - epoch: " << i << "<<<<<<<" << std::endl;
        resnet50_ptr->eval();
        float val_loss = 0;
        int val_correct = 0;
        int val_batch_num = 0;
        for (const std::vector<torch::data::Example<>> &batch : *val_loader)
        { // 返回vector包装的批次数据
            std::vector<torch::Tensor> imgs;
            std::vector<torch::Tensor> labels;
            for (torch::data::Example<> data_label : batch)
            {
                imgs.push_back(data_label.data);
                labels.push_back(data_label.target);
            }
            torch::Tensor imgs_t = torch::stack(torch::TensorList(imgs)).to(device);
            torch::Tensor labels_t = torch::stack(torch::TensorList(labels)).to(device);
            torch::Tensor pred = resnet50_ptr->forward(imgs_t);
            torch::Tensor loss = torch::cross_entropy_loss(pred, labels_t);
            // top1 准确率
            torch::Tensor pred_label = torch::argmax(pred, 1);
            val_loss = val_loss + loss.item<float>();
            val_correct = val_correct + torch::sum(pred_label == labels_t).item<int>();
            val_batch_num++;
            std::cout << " epoch: " << i
                      << " batch_num : " << val_batch_num << " / " << val_batch_count
                      << " batch_size : " << batch.size()
                      << " val_acc: " << torch::sum(pred_label == labels_t).item<int>() / labels_t.size(0)
                      << " batch_loss: " << loss.item<float>() << std::endl;
        }
        float _val_acc = val_correct / float(val_dataset_ptr->size().value());
        std::cout << ">>>>>>> val stage - result - epoch: " << i << "<<<<<<<" << std::endl;
        std::cout << " epoch: " << i
                  << " val accuracy: " << _val_acc
                  << " val loss: " << val_loss / val_batch_num << std::endl;
        // 保存模型
        if (val_acc < _val_acc)
        {
            std::cout << ">>> The best accuracy :" << _val_acc << std::endl;
            val_acc = _val_acc;
            torch::serialize::OutputArchive archive_out;                             // 创建输出archive
            resnet50_ptr->save(archive_out);                                         // 将模型参数保存到archive
            archive_out.save_to("../logs/alzheimer_resnet50/alzheimer_resnet50.pt"); // 将archive保存到文件
            std::cout << ">>> save model success !" << std::endl;
        }
    }
}

void jit_script_test()
{
    std::shared_ptr<torch::jit::Module> resnet50_ptr = std::make_shared<torch::jit::Module>(torch::jit::load("../logs/jit_script_model/script_resnet50.pt"));
    resnet50_ptr->eval();
    std::cout << ">>> jit script model load success !" << std::endl;
    for (int i = 0; i < 10; i++)
    {
        torch::Tensor input = torch::rand({1, 3, 224, 224});
        // std::vector<IValue> 表示了一个装有 torch::jit::IValue 对象的向量，因此可以容纳各种类型的输入数据。
        // torch::jit::IValue 类型是 PyTorch 中用于表示任意数据类型的一个抽象类，它可以表示张量、标量、列表、字典等各种数据类型。
        torch::Tensor output = resnet50_ptr->forward({input}).toTensor();
        std::cout << ">>> jit script model output: " << output.slice(1, 0, 5) << std::endl;
    }
    // 虽然接收参数是std::vector<torch::jit::IValue>，但实际要根据导出的模型参数来传参
    // for (int i = 0; i < 10; i++)
    // {
    //     std::vector<torch::Tensor> input = {torch::rand({3, 224, 224}), torch::rand({3, 224, 224})};
    //     torch::Tensor output = resnet50_ptr->forward({input}).toTensor();
    //     std::cout << ">>> jit script model output: " << output.slice(1, 0, 5) << std::endl;
    // }
}