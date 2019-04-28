cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(Word2Vec-Skipgram-Torch)

find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "-Wall -g3")

add_executable(Word2Vec-Skipgram-Torch Word2Vec-Skipgram-Torch\(Softmax\).cpp NLPDataset.cpp)
target_compile_features(Word2Vec-Skipgram-Torch PUBLIC cxx_range_for)
target_link_libraries(Word2Vec-Skipgram-Torch ${TORCH_LIBRARIES})

