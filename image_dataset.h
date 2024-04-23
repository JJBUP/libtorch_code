#include <torch/torch.h>
// 查看源码可知：<>的作用是为了将ImageDataset传递给Dataset模板类以便
class ImageDataset : public torch::data::Dataset<ImageDataset>
{
public:
    ImageDataset(const std::string &root_dir, const std::map<std::string, int> &class_name, std::string mode); //,const std::vector<std::string> &transforms)
    torch::data::Example<> get(size_t index);
    std::optional<size_t> size() const override;
    void recursive_rglob(const std::string &directory_path, std::vector<std::string> *const paths_ptr);

private:
    std::string rd;
    std::map<std::string, int> cls;
    std::vector<std::string> paths;
    std::string md;
};