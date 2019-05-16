
#include "NLPDataset.h"

#include <torch/data/example.h>
#include <torch/types.h>
#include <c10/util/Exception.h>

#include <cstddef>
#include <fstream>
#include <string>
#include <vector>

namespace torch {
namespace data {
namespace datasets {
namespace {
constexpr uint32_t kInit = 3;
constexpr uint32_t kLast = 1;

} // namespace

  NLP::NLP(std::vector<std::pair<std::string, std::string>> data_seq) {

    std::vector<int64_t> concat_input_words;
    std::vector<int64_t> concat_output_words;
    std::vector<int64_t> concat_target_words;

  int index=0;
  for( auto c : std::string("SEPabcdefghijklmnopqrstuvwxyz") ) {
    std::string str = std::string(1, c);
    words.insert(str);
    word_index[str] = index;
    index_word[index] = str;
    index++;
  }

  uint64_t input_max = 0;
  uint64_t output_max = 0;

  for( auto data : data_seq ) {
    auto input = data.first;
    auto output = data.second;
    input_max = std::max(input_max, input.length());
    output_max = std::max(output_max, output.length());
  }

  for( auto data : data_seq ) {

    {
      int i = 0;
      for( auto c : data.first ) {
        std::string str = std::string(1, c);
        concat_input_words.push_back(word_index[str]);
        i++;
      }
      for( ; i < input_max ; i++ ) {
        concat_input_words.push_back(word_index["P"]);
      }
    }

    {
      int i = 0;
      concat_output_words.push_back(word_index["S"]);
      for( auto c : data.second ) {
        std::string str = std::string(1, c);
        concat_output_words.push_back(word_index[str]);
        concat_target_words.push_back(word_index[str]);
        i++;
      }
      for( ; i < output_max ; i++ ) {
        concat_output_words.push_back(word_index["P"]);
        concat_target_words.push_back(word_index["P"]);
      }
      concat_target_words.push_back(word_index["E"]);
    }
  }

  const uint64_t count = data_seq.size();
  torch::Tensor input = torch::empty({count, input_max}, torch::kInt64);
  torch::Tensor output = torch::empty({count, 1, output_max+1}, torch::kInt64);
  torch::Tensor target = torch::empty({count, 1, output_max+1}, torch::kInt64);
  std::memcpy(input.data_ptr(), concat_input_words.data(), input.numel() * sizeof(int64_t));
  std::memcpy(output.data_ptr(), concat_output_words.data(), output.numel() * sizeof(int64_t));
  std::memcpy(target.data_ptr(), concat_target_words.data(), target.numel() * sizeof(int64_t));
  input_   = input;
  targets_ = torch::cat({output, target}, 1);
}

int64_t NLP::getClassNumber() {
  return words.size();
}

Example<> NLP::get(size_t index) {
  return {input_[index], targets_[index]};
}

optional<size_t> NLP::size() const {
  return input_.size(0);
}

const Tensor& NLP::input() const {
  return input_;
}

const Tensor& NLP::targets() const {
  return targets_;
}

const std::string& NLP::index_to_string(int64_t index) {
  return index_word[index];
}


} // namespace datasets
} // namespace data
} // namespace torch
