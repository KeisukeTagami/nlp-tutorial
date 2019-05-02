cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(TextLSTM_Torch)

find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "-Wall -g3")


add_executable(TextLSTM_Torch TextLSTM_Torch.cpp NLPDataset.cpp)
target_compile_features(TextLSTM_Torch PUBLIC cxx_range_for)
target_link_libraries(TextLSTM_Torch ${TORCH_LIBRARIES})

