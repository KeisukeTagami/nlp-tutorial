
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


NLP::NLP(std::vector<std::pair<std::string,std::string>> sentences) {

  std::vector<int64_t> concat_input_words;
  std::vector<int64_t> concat_output_words;
  std::vector<int64_t> concat_target_words;

  int index=0;

  auto buildIndices = [&](std::string sentence, std::function<void(int64_t)> yeild) {

                        char delim = ' ';
                        std::string word;
                        std::stringstream ss(sentence);
                        while (getline(ss, word, delim)) {
                          if (!word.empty()) {

                            if( words.insert(word).second ) {
                              word_index[word] = index;
                              index_word[index] = word;
                              index++;
                            }
                            yeild(word_index[word]);
                          }
                        }

                      };

  int64_t max_length = 0;

  buildIndices("[PAD] [CLS] [SEP] [MASK]", [&](int64_t index){} );
  for( auto sentence : sentences ) {
    int64_t input_length = 0;
    int64_t output_length = 0;
    buildIndices(sentence.first, [&](int64_t index){ input_length++;} );
    buildIndices(sentence.second, [&](int64_t index){ output_length++; } );
    max_length = std::max(max_length, input_length);
    max_length = std::max(max_length, output_length);
  }

  for( auto sentence : sentences ) {
    int64_t input_length = 0;
    int64_t output_length = 0;

    concat_output_words.push_back(word_index["S"]);
    buildIndices(sentence.first, [&](int64_t index){ concat_input_words.push_back(index); input_length++;} );
    buildIndices(sentence.second, [&](int64_t index){
                                          concat_output_words.push_back(index);
                                          concat_target_words.push_back(index);
                                          output_length++;
                                        } );

    for( ; input_length < max_length ; input_length++ ) {
      concat_input_words.push_back(word_index["P"]);
    }

    for( ; output_length < max_length ; output_length++ ) {
      concat_output_words.push_back(word_index["P"]);
      concat_target_words.push_back(word_index["P"]);
    }
    concat_input_words.push_back(word_index["P"]);
    concat_target_words.push_back(word_index["E"]);
  }

  const int64_t count = sentences.size();
  torch::Tensor input = torch::empty({count, max_length+1}, torch::kInt64);
  torch::Tensor output = torch::empty({count, max_length+1}, torch::kInt64);
  torch::Tensor target = torch::empty({count, max_length+1}, torch::kInt64);
  std::memcpy(input.data_ptr(), concat_input_words.data(), input.numel() * sizeof(int64_t));
  std::memcpy(output.data_ptr(), concat_output_words.data(), output.numel() * sizeof(int64_t));
  std::memcpy(target.data_ptr(), concat_target_words.data(), target.numel() * sizeof(int64_t));
  input_   = input;
  output_  = output;
  targets_ = target;
}

int64_t NLP::getClassNumber() {
  return words.size();
}

RNNExample NLP::get(size_t index) {
    return {{ input_[index], output_[index]} , targets_[index]};
}

optional<size_t> NLP::size() const {
  return input_.size(0);
}

const Tensor& NLP::input() const {
  return input_;
}

const Tensor& NLP::output() const {
  return output_;
}

const Tensor& NLP::targets() const {
  return targets_;
}

const std::string& NLP::index_to_string(int64_t index) {
  return index_word[index];
}

Tensor NLP::strings_to_tensor(std::vector<std::string> sentences) {

  std::vector<int64_t> concat_output_words;

  auto processSentence = [&](std::string sentence, std::function<void(std::string&)> yeild) {
                           char delim = ' ';
                           std::string word;
                           std::stringstream ss(sentence);
                           while (getline(ss, word, delim)) {
                             if (!word.empty()) {
                               yeild(word);
                             }
                           }
                         };

  int64_t output_max = 0;
  for( auto sentence : sentences ) {
    int64_t output_length = 0;
    processSentence(sentence, [&](std::string& word) { output_length++; } );
    output_max = std::max(output_max, output_length);
  }

  for( auto sentence : sentences ) {
    int64_t output_length = 0;
    processSentence(sentence, [&](std::string& word) {
                                concat_output_words.push_back(word_index[word]);
                                output_length++;
                              } );
    for( ; output_length < output_max ; output_length++ ) {
      concat_output_words.push_back(word_index["P"]);
    }
  }

  const int64_t count = sentences.size();
  torch::Tensor output = torch::empty({count, output_max}, torch::kInt64);
  std::memcpy(output.data_ptr(), concat_output_words.data(), output.numel() * sizeof(int64_t));
  return torch::one_hot(output, getClassNumber()).to(torch::kFloat);
}


} // namespace datasets
} // namespace data
} // namespace torch
