
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
constexpr uint32_t kSentence = 3;

} // namespace

NLP::NLP(std::vector<std::string> sentences,
         std::vector<int64_t> labels) {

  std::vector<int64_t> concat_sentences;

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
  input_ = torch::empty({count, kSentence}, torch::kInt64);
  std::memcpy(input_.data_ptr(), concat_sentences.data(), input_.numel() * sizeof(int64_t));
  targets_ = torch::empty({count, 1}, torch::kInt64);
  std::memcpy(targets_.data_ptr(), labels.data(), targets_.numel() * sizeof(int64_t));
  targets_ = targets_.view({-1});
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

const int64_t NLP::string_to_index(std::string string) {
  return word_index[string];
}


} // namespace datasets
} // namespace data
} // namespace torch
