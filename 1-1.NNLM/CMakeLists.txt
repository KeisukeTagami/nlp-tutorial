cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(NNLM_Torch)

find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "-Wall -g3")

add_executable(NNLM_Torch NNLM_Torch.cpp NLPDataset.cpp)
target_compile_features(NNLM_Torch PUBLIC cxx_range_for)
target_link_libraries(NNLM_Torch ${TORCH_LIBRARIES})

