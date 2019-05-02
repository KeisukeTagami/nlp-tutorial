
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

  NLP::NLP(std::vector<std::string> words_) {

  std::vector<int64_t> concat_words;

  int index=0;
  for( auto c : std::string("abcdefghijklmnopqrstuvwxyz") ) {
    std::string str = std::string(1, c);
    words.insert(str);
    word_index[str] = index;
    index_word[index] = str;
    index++;
  }

  for( auto word : words_ ) {
    for( auto c : word ) {
      std::string str = std::string(1, c);
      concat_words.push_back(word_index[str]);
    }
  }

  const auto count = words_.size();
  auto tensor = torch::empty({count, kInit + kLast}, torch::kInt64);
  std::memcpy(tensor.data_ptr(), concat_words.data(), tensor.numel() * sizeof(int64_t));
  input_   = tensor.slice(1, 0, kInit);
  targets_ = tensor.slice(1, kInit, kInit + kLast);
  targets_ = targets_.view({-1});
}

int64_t NLP::getClassNumber() {
  return words.size();
}

Example<> NLP::get(size_t index) {
  auto one_hot = torch::one_hot(input_[index], getClassNumber()).to(torch::kFloat);
  return {one_hot, targets_[index]};
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
