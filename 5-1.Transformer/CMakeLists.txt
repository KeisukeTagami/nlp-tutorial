cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(Transformer)

find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "-Wall -g3")

add_executable(Transformer_Torch Transformer_Torch.cpp NLPDataset.cpp)
target_compile_features(Transformer_Torch PUBLIC cxx_range_for)
target_link_libraries(Transformer_Torch ${TORCH_LIBRARIES} ${CMAKE_DL_LIBS})

