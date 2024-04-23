#include <torch/torch.h>
#include <string>
// 查看源码可知：<>的作用是为了将ImageDataset传递给Dataset模板类以便
class ImageDataset : public torch::data::Dataset<ImageDataset>
{
public:
    std::map<std::string, int> cls;
    std::vector<std::string> tfs;
    std::filesystem::path rd;
    std::vector<std::filesystem::path> paths;
    ImageDataset(const std::string &root_dir, const std::map<std::string, int> &class_name, const std::vector<std::string> &transforms);
    torch::data::Example<> get(size_t index);
    torch::optional<size_t> size();

private:
    std::vector<std::string> m_root_dir;
};