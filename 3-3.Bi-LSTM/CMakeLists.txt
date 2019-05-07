cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(Bi_LSTM_Torch)

find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "-Wall -g3")


add_executable(Bi_LSTM_Torch Bi_LSTM_Torch.cpp NLPDataset.cpp)
target_compile_features(Bi_LSTM_Torch PUBLIC cxx_range_for)
target_link_libraries(Bi_LSTM_Torch ${TORCH_LIBRARIES})

