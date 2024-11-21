cmake_minimum_required(VERSION 3.29)
project(LayerNormPlugin LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 14)

#find_package(CUDA REQUIRED)

#add_executable(LayerNormPlugin
#        LayerNormPlugin.cu
#        LayerNormPlugin.h
#)
add_library(LayerNormPlugin SHARED
        LayerNormPlugin.cu
        LayerNormPlugin.h
)

file(GLOB plugin "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/lib/x64/*.lib")
target_link_libraries(LayerNormPlugin
        ${plugin}
)

# 忽略冲突的默认库
target_link_options(LayerNormPlugin PRIVATE "/NODEFAULTLIB:LIBCMT")