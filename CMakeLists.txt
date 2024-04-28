cmake_minimum_required(VERSION 3.10)
project(MyProject)

set(CMAKE_BUILD_TYPE Debug) # 设置编译类型为 Debug,默认为 Release

set(CMAKE_CXX_STANDARD 17)
# 设置 CMake 搜索库的路径
# set(CMAKE_PREFIX_PATH "/home/jjb/libtorch/share/cmake/Torch")
# set(CMAKE_PREFIX_PATH "/home/jjb/opencv-4.9.0-compile/")

find_package(Torch REQUIRED CONFIG PATHS /home/jjb/libtorch)
find_package(OpenCV REQUIRED CONFIG PATHS /home/jjb/opencv-4.9.0-compile/)

file(GLOB SOURCES "./*.cpp")
add_executable(SystemUpdate ${SOURCES})

# 在 CMake 中包含头文件路径
include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

# 链接 PyTorch 库
target_link_libraries(SystemUpdate ${TORCH_LIBRARIES} ${OpenCV_LIBRARIES})