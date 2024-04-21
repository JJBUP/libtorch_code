cmake_minimum_required(VERSION 3.10)
project(MyProject)

set(CMAKE_BUILD_TYPE Debug) # 设置编译类型为 Debug,默认为 Release

set(CMAKE_CXX_STANDARD 17)
# 设置 CMake 搜索库的路径
set(CMAKE_PREFIX_PATH "/home/jjb/libtorch/")

find_package(Torch REQUIRED)

add_executable(libtorch_test libtorch_test.cpp linear_regression.cpp)

# 在 CMake 中包含头文件路径
include_directories(${TORCH_INCLUDE_DIRS})

# 链接 PyTorch 库
target_link_libraries(libtorch_test ${TORCH_LIBRARIES})