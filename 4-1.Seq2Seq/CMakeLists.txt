cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(Seq2Seq_Torch)

find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "-Wall -g3")

add_executable(Seq2Seq_Torch Seq2Seq_Torch.cpp NLPDataset.cpp)
target_compile_features(Seq2Seq_Torch PUBLIC cxx_range_for)
target_link_libraries(Seq2Seq_Torch ${TORCH_LIBRARIES} ${CMAKE_DL_LIBS})

