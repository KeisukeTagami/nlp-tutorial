cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(TextRNN_Torch)

find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "-Wall -g3")


add_executable(TextRNN_Torch TextRNN_Torch.cpp NLPDataset.cpp)
target_compile_features(TextRNN_Torch PUBLIC cxx_range_for)
target_link_libraries(TextRNN_Torch ${TORCH_LIBRARIES})

