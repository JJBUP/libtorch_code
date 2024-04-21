#include <torch/torch.h>
#include <iostream>
#include <vector>
#include "linear_regression.h"
void tensor_create();
void tensor_index();
void tensor_operation();
void auto_grad();
void simulate_linear_regression();
void simulate_img_classification();
int main()
{
    // tensor_create();
    // tensor_index();
    // tensor_operation();
    simulate_linear_regression();
    // simulate_linear_regression();
    // simulate_img_classification();
}

// 创建tensor
void tensor_create()
{
    // 基础数据创建
    torch::Tensor a = torch::zeros({2, 3});
    torch::Tensor b = torch::ones({2, 3});
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
    torch::Tensor b = a.index({0});                                                                            // 对应python中的[0]
    torch::Tensor c = a.index({0, 1});                                                                         // 对应python中的[0,1]
    torch::Tensor d = a.index({0, 1, -1});                                                                     // 对应python中的[0,1,-1]
    torch::Tensor e = a.index({torch::indexing::Slice(), 1, 2});                                               // Slice(int start,int end)类用于设定单一维度切片范围,Slice()对应python中的:
    torch::Tensor f = a.index({torch::indexing::Slice(0, -1), 1, 2});                                          // Slice(int start,int end)类用于设定单一维度切片范围,Slice()对应python中的:
    torch::Tensor g = a.index({"...", 2});                                                                     // "..."对应python中的...
    torch::Tensor h = a.index({torch::indexing::Ellipsis, 2});                                                 // Ellipsis对应python中的...
    torch::Tensor i = a.index({torch::indexing::None});                                                        // None对应python中的None,增加一个新的维度
    torch::Tensor j = torch::rand({2, 3}).to(torch::kBool).index({"...", true});                               // kBool为ScalarType基础张量类型,true为python中的True
    torch::Tensor k = a.index({torch::tensor({0, 0, 0}), torch::tensor({1, 1, 1}), torch::tensor({2, 2, 2})}); // int类型tensor作为索引
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
    std::cout << a.requires_grad() << std::endl;      // requires_grad() 函数,获取是否需要梯度,

    // 修改数据
    std::cout << a.to(torch::kCUDA) << std::endl;                // to() 函数,获取指定设备类型张量
    std::cout << a.to(torch::Device(torch::kCUDA)) << std::endl; // to() 函数,获取指定设备类型张量
    std::cout << a.to(torch::Device("cuda:0")) << std::endl;     // to() 函数,获取指定设备类型张量
    std::cout << a.to(torch::kCPU) << std::endl;                 // to() 函数,获取指定设备类型张量
    std::cout << a.detach() << std::endl;                        // detach() 函数,获取不带梯度张量
    std::cout << a.cuda() << std::endl;                          // cuda() 函数,获取cuda张量
    std::cout << a.data() << std::endl;                          // data() 函数,获取数据指针
    std::cout << a.grad() << std::endl;                          // grad() 函数,获取梯度
    std::cout << a[0][0].item<float>() << std::endl;             // item() 函数,获取数据
    std::cout << a[0][0].item() << std::endl;                    // item() 函数,获取数据
    std::cout << a.isnan() << std::endl;                         // isnan() 函数,获取是否是NaN
    std::cout << a.isinf() << std::endl;                         // isinf() 函数,获取是否是无穷大
    std::cout << a.isfinite() << std::endl;                      // isfinite() 函数,获取是否是有穷
    std::cout << a.clone() << std::endl;                         // clone() 函数,获取克隆张量
    std::cout << a.flatten() << std::endl;                       // flatten() 函数,获取展平张量
    std::cout << a.view({3, 2}) << std::endl;                    // view() 函数,获取重塑张量
    std::cout << a.reshape({3, 2}) << std::endl;                 // reshape() 函数,获取重塑张量
    std::cout << a.transpose(0, 1) << std::endl;                 // transpose() 函数,获取转置张量
    std::cout << a.permute({1, 0}) << std::endl;                 // permute() 函数,获取转置张量
    std::cout << a.unsqueeze(0) << std::endl;                    // unsqueeze() 函数,获取增加维度张量
    std::cout << a.squeeze(0) << std::endl;                      // squeeze() 函数,获取删除维度张量
    std::cout << a.cumsum(0) << std::endl;                       // cumsum() 函数,获取累加张量
    std::cout << a.cumprod(0) << std::endl;                      // cumprod() 函数,获取累乘张量
    std::cout << std::get<0>(a.topk(1, -1)) << std::endl;        // topk() 函数,获取topk张量数值
    std::cout << std::get<1>(a.topk(1, -1)) << std::endl;        // topk() 函数,获取topk张量索引
    std::cout << a.contiguous() << std::endl;                    // contiguous() 函数,获取连续张量
    std::cout << torch::cat({a, a}, 1) << std::endl;             // cat() 函数,获取拼接张量
    // std::cout << a.to_sparse() << std::endl;                     // to_sparse() 函数,获取稀疏张量
    // std::cout << a.to_dense() << std::endl;                      // to_dense() 函数,获取稠密张量
    // std::cout << a.to_sparse_csr() << std::endl;                 // to_sparse_csr() 函数,获取稀疏张量
    // std::cout << a.to_sparse_csc() << std::endl;                 // to_sparse_csc() 函数,获取稀疏张量

    // 数学计算
    std::tuple svd_data = a.svd();
    std::cout << std::get<0>(svd_data) << std::get<1>(svd_data) << std::get<2>(svd_data) << std::endl; // svd() 函数,获取SVD张量
    std::cout << a.t() << std::endl;
    std::cout << a.argmax() << std::endl;
    std::cout << a.max() << std::endl;
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
    d = c * (a * b);
    e = d.sum();
    d.retain_grad(); // 保留梯度
    e.retain_grad(); // 保留梯度
    e.backward();    // 反向传播
    std::cout << "a.grad:" << a.grad() << std::endl;
    std::cout << "b.grad:" << b.grad() << std::endl;
    std::cout << "c.grad:" << c.grad() << std::endl;
    std::cout << "d.grad:" << d.grad() << std::endl; // 非叶子节点
    std::cout << "e.gard:" << e.grad() << std::endl; // 非叶子节点
}

// 模拟线性回归
void simulate_linear_regression()
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
void simulate_img_classification()
{
    // 输入张量
    torch::Tensor img = torch::randn({1, 3, 224, 224});
    // 卷积核张量
    torch::Tensor w = torch::randn({64, 3, 3, 3}, torch::requires_grad(false));
    // 输出张量

}
