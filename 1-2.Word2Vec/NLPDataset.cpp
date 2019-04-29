
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

} // namespace

  NLP::NLP(std::vector<std::string> sentences) {

  int index=0;
  char delim = ' ';
  std::string word;

  std::vector<int64_t> skipgram_target;
  std::vector<int64_t> skipgram_context_before;
  std::vector<int64_t> skipgram_context_after;

  for( auto sentence : sentences ) {
    std::stringstream ss(sentence);
    std::vector<int64_t> word_indices;

    while (getline(ss, word, delim)) {

      if (!word.empty()) {

        if( words.insert(word).second ) {
          word_index[word] = index;
          index_word[index] = word;
          index++;
        }

        word_indices.push_back(word_index[word]);
      }

    }

    for( auto idx = 1; idx < static_cast<int64_t>(word_indices.size()) - 1; idx++ ) {
      skipgram_target.push_back(word_indices[idx]);
      skipgram_context_before.push_back(word_indices[idx-1]);
      skipgram_context_after.push_back(word_indices[idx+1]);
    }
  }

  const auto count = static_cast<int64_t>(skipgram_target.size());

  auto input = torch::empty({count}, torch::kInt64);
  std::memcpy(input.data_ptr(), skipgram_target.data(), input.numel() * sizeof(int64_t));

  auto context_before = torch::empty({count}, torch::kInt64);
  std::memcpy(context_before.data_ptr(), skipgram_context_before.data(), context_before.numel() * sizeof(int64_t));

  auto context_after = torch::empty({count}, torch::kInt64);
  std::memcpy(context_after.data_ptr(), skipgram_context_after.data(), context_after.numel() * sizeof(int64_t));

  input_ = at::cat({input, input}, 0);
  targets_ = at::cat({context_before, context_after}, 0);

  std::cout << input_ << std::endl;
  std::cout << targets_ << std::endl;
}

int64_t NLP::getClassNumber() {
  return words.size();
}

Example<> NLP::get(size_t index) {
  auto eye = torch::eye(getClassNumber());
  return {eye[input_[index]], targets_[index]};
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
