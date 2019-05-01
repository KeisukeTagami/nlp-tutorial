cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(TextCNN)

find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "-Wall -g3")

add_executable(TextCNN_Torch TextCNN_Torch.cpp NLPDataset.cpp)
target_compile_features(TextCNN_Torch PUBLIC cxx_range_for)
target_link_libraries(TextCNN_Torch ${TORCH_LIBRARIES})

