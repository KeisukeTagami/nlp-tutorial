cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(BERT_Torch)

find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "-Wall -g3")

add_executable(BERT_Torch BERT_Torch.cpp NLPDataset.cpp)
target_compile_features(BERT_Torch PUBLIC cxx_range_for)
target_link_libraries(BERT_Torch ${TORCH_LIBRARIES} ${CMAKE_DL_LIBS})

