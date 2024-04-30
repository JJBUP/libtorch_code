cmake_minimum_required(VERSION 3.10)
project(MyProject)

set(CMAKE_BUILD_TYPE Release) # 设置编译类型为 Debug,默认为 Release

set(CMAKE_CXX_STANDARD 17)
# 设置 CMake 搜索库的路径
# set(CMAKE_PREFIX_PATH "/home/jjb/libtorch/share/cmake/Torch")
# set(CMAKE_PREFIX_PATH "/home/jjb/opencv-4.9.0-compile/")

find_package(Torch REQUIRED CONFIG PATHS /home/jjb/libtorch)
find_package(OpenCV REQUIRED CONFIG PATHS /home/jjb/opencv-4.9.0-compile/)
# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src_model) # 为所有目标目添加头文件目录


# 搜索指定目录cpp文件
file(GLOB SOURCES1 ./*.cpp)
file(GLOB SOURCES2 ./src_model/*.cpp)
file(GLOB SOURCES3 ./src_dataset/*.cpp)
# 添加头文件搜索路径
# include_directories和target_include_directories的区别在于，
# include_directories是针对所有目标都生效，而target_include_directories是针对某个目标生效。
include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

# 链接cpp到目标
add_executable(libtorch_test ${SOURCES1} ${SOURCES2} ${SOURCES3})
# 链接头文件到目标
target_include_directories(libtorch_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} 
                                                ${CMAKE_CURRENT_SOURCE_DIR}/src_model
                                                ${CMAKE_CURRENT_SOURCE_DIR}/src_dataset)

# 链接 第三方库
target_link_libraries(libtorch_test ${TORCH_LIBRARIES} ${OpenCV_LIBRARIES})