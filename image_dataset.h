#include <torch/torch.h>
// 查看源码可知：<>的作用是为了将ImageDataset传递给Dataset模板类以便
class ImageDataset : public torch::data::Dataset<ImageDataset>
{
public:
    ImageDataset(const std::string &root_dir, const std::map<std::string, int> &class_name, std::string mode = "train"); //,const std::vector<std::string> &transforms)
    torch::data::Example<> get(size_t index);
    std::optional<size_t> size();
    void recursive_rglob(const std::filesystem::path &directory_path, std::vector<std::filesystem::path> *const paths);

private:
    std::filesystem::path rd;
    std::map<std::string, int> cls;
    std::vector<std::filesystem::path> paths;
    std::string md;
};