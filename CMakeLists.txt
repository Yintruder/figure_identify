cmake_minimum_required(VERSION 3.10) # 设置CMake最低版本要求
project(FigureProject) # 设置项目名称

set(CMAKE_CXX_STANDARD 14) # 设置C++标准

# 添加可执行文件
# 第一个参数是可执行文件的名称
# 第二个参数是你的源文件


find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 添加可执行文件
# 第一个参数是可执行文件的名称
# 第二个参数是你的源文件
add_executable(FigureProject figure.cpp)
target_link_libraries(FigureProject ${OpenCV_LIBS})