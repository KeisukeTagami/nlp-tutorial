
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
constexpr uint32_t kInit = 2;
constexpr uint32_t kLast = 1;

} // namespace

  NLP::NLP(std::vector<std::string> sentences) {

  std::vector<int64_t>           concat_sentences;

  int index=0;
  char delim = ' ';
  std::string word;

  for( auto sentence : sentences ) {
    std::stringstream ss(sentence);

    while (getline(ss, word, delim)) {
      if (!word.empty()) {

        if( words.insert(word).second ) {
          word_index[word] = index;
          index_word[index] = word;
          index++;
        }
        concat_sentences.push_back(word_index[word]);
      }
    }
  }

  const auto count = sentences.size();
  // auto tensor = torch::from_blob(concat_sentences.data(), {count, kInit + kLast});
  auto tensor = torch::empty({count, kInit + kLast}, torch::kInt64);
  std::memcpy(tensor.data_ptr(), concat_sentences.data(), tensor.numel() * sizeof(int64_t));
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
