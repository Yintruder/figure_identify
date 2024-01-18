cmake_minimum_required(VERSION 3.10)
project(CodeRecognizerProject)

# 设置C++标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# OpenCV的查找和包含
find_package(OpenCV 4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 包含头文件目录
include_directories(include)

# 添加src目录下的源文件
aux_source_directory(src SRC_LIST)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})
# 生成可执行文件
add_executable(CodeRecognizerExample example/main.cpp ${SRC_LIST})

# 链接OpenCV库
target_link_libraries(CodeRecognizerExample ${OpenCV_LIBS})
