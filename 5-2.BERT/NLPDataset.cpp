
#include "NLPDataset.h"

#include <torch/data/example.h>
#include <torch/types.h>
#include <c10/util/Exception.h>

#include <cstddef>
#include <fstream>
#include <string>
#include <vector>
#include <regex>
#include <random>

namespace torch {
namespace data {
namespace datasets {

NLP::NLP(std::vector<std::string> sentences) {

  int index=0;

  auto buildIndices = [&](std::string s, std::function<void(int64_t)> yeild, std::function<void()> punctuate, bool preprocess=true) {

                        if( preprocess ){
                          std::regex re( R"([.,!?\\-])" ) ;
                          s = std::regex_replace( s, re, "" ) ;
                          std::transform(s.begin(), s.end(), s.begin(), ::tolower);
                        }

                        std::regex sp{"\n"};
                        std::sregex_token_iterator begin{s.begin(), s.end(), sp, -1}, end;

                        std::for_each(begin, end,
                                      [&](std::string sentence) {

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

                                        punctuate();
                                      });
                      };

  int64_t max_length = 0;

  buildIndices("[PAD] [CLS] [SEP] [MASK]", [&](int64_t index){}, [](){} , false);
  for( auto sentence : sentences ) {
    int64_t input_length = 0;
    buildIndices(sentence,
                 [&](int64_t index){
                   input_length++;
                 },
                 [&](){
                   max_length = std::max(max_length, input_length);
                   input_length = 0;
                 } );
  }

  max_length = max_length * 2 + 3;
  std::vector<std::vector<int64_t>> token_list;
  std::vector<bool> is_continuous;

  for( auto sentence : sentences ) {
    std::vector<int64_t> token;
    buildIndices(sentence,
                 [&](int64_t index){ token.push_back(index); },
                 [&](){
                   token_list.push_back(token);
                   token.clear();
                   is_continuous.push_back(true);
                 }
                 );
    is_continuous.pop_back();
    is_continuous.push_back(false);
  }


  std::vector<int64_t> concat_input_words;
  std::vector<int64_t> _input_ids, _segment_ids, _masked_tokens, _masked_pos, _isNext;

  int64_t positive = 0;
  int64_t negative = 0;
  int64_t batch_size = 6;
  int64_t max_pred = 5;

  std::random_device rnd;
  std::mt19937 mt(rnd());
  std::uniform_int_distribution<> rand_input(0, token_list.size()-1);
  std::uniform_int_distribution<> rand_word(4, getClassNumber()-1);

  std::normal_distribution<> rand_norm(0.0, 1.0);

  while( positive != batch_size/2 || negative != batch_size/2 ) {

    int64_t tokens_a_index = rand_input(mt);
    int64_t tokens_b_index = rand_input(mt);

    std::vector<int64_t> tokens_a = token_list[tokens_a_index];
    std::vector<int64_t> tokens_b = token_list[tokens_b_index];

    std::vector<int64_t> input_ids;
    input_ids.insert(input_ids.end(), {word_index["[CLS]"]});
    input_ids.insert(input_ids.end(), tokens_a.begin(), tokens_a.end());
    input_ids.insert(input_ids.end(), {word_index["[SEP]"]});
    input_ids.insert(input_ids.end(), tokens_b.begin(), tokens_b.end());
    input_ids.insert(input_ids.end(), {word_index["[SEP]"]});

    std::vector<int64_t> segment_ids;
    for(int i = 0; i < 1 + tokens_a.size() + 1; i++ )
      segment_ids.push_back(0);
    for(int i = 0; i < tokens_b.size() + 1; i++ )
      segment_ids.push_back(1);

    //  MASK LM
    int64_t n_pred = std::min(max_pred, std::max((int64_t)1, (int64_t)std::round(input_ids.size() * 0.15) )); // 15 % of tokens in one sentence
    std::vector<int64_t> cand_maked_pos;

    int i = 0;
    for( int token : input_ids ) {
      if( token != word_index["[CLS]"] && token != word_index["[SEP]"] )
        cand_maked_pos.push_back(i);
      i++;
    }

    std::shuffle(cand_maked_pos.begin(), cand_maked_pos.end(), mt);

    std::vector<int64_t> masked_tokens, masked_pos;

    i = 0;
    for( auto pos : cand_maked_pos ) {
      masked_pos.push_back(pos);
      masked_tokens.push_back(input_ids[pos]);
      if (rand_norm(mt) <= 0.8) { // 80%
        input_ids[pos] = word_index["[MASK]"]; // # make mask
      } else if ( rand_norm(mt) <= 0.5 ) { // 10%
        index = rand_word(mt); // random index in vocabulary
        input_ids[pos] = word_index[index_word[index]]; // # replace
      }
      if( ++i >= n_pred ) break;
    }

    // Zero Paddings
    auto n_pad = max_length - input_ids.size();
    for( int i = 0 ; i < n_pad ; i++ ) {
      input_ids.push_back(word_index["[PAD]"]);
      segment_ids.push_back(word_index["[PAD]"]);
    }

    // Zero Padding (100% - 15%) tokens
    n_pad = max_pred - n_pred;
    for( int i = 0 ; i < n_pad ; i++ ) {
      masked_tokens.push_back(0);
      masked_pos.push_back(0);
    }

    if( tokens_a_index + 1 == tokens_b_index && positive < batch_size/2 && is_continuous[tokens_a_index] ) {
      _input_ids.insert(_input_ids.end(), input_ids.begin(), input_ids.end() );
      _segment_ids.insert(_segment_ids.end(), segment_ids.begin(), segment_ids.end() );
      _masked_tokens.insert(_masked_tokens.end(), masked_tokens.begin(), masked_tokens.end());
      _masked_pos.insert(_masked_pos.end(), masked_pos.begin(), masked_pos.end());
      _isNext.push_back(1);
      positive += 1;
    } else if( tokens_a_index + 1 != tokens_b_index && negative < batch_size/2 ) {
      _input_ids.insert(_input_ids.end(), input_ids.begin(), input_ids.end() );
      _segment_ids.insert(_segment_ids.end(), segment_ids.begin(), segment_ids.end() );
      _masked_tokens.insert(_masked_tokens.end(), masked_tokens.begin(), masked_tokens.end());
      _masked_pos.insert(_masked_pos.end(), masked_pos.begin(), masked_pos.end());
      _isNext.push_back(0);
      negative += 1;
    }
  }

  torch::Tensor input_ids = torch::empty({batch_size, max_length}, torch::kInt64);
  torch::Tensor segment_ids = torch::empty({batch_size, max_length}, torch::kInt64);
  torch::Tensor masked_tokens = torch::empty({batch_size, max_pred}, torch::kInt64);
  torch::Tensor masked_pos = torch::empty({batch_size, max_pred}, torch::kInt64);
  torch::Tensor isNext     = torch::empty({batch_size, 1}, torch::kInt64);
  std::memcpy(input_ids.data_ptr(), _input_ids.data(), input_ids.numel() * sizeof(int64_t));
  std::memcpy(segment_ids.data_ptr(), _segment_ids.data(), segment_ids.numel() * sizeof(int64_t));
  std::memcpy(masked_tokens.data_ptr(), _masked_tokens.data(), masked_tokens.numel() * sizeof(int64_t));
  std::memcpy(masked_pos.data_ptr(), _masked_pos.data(), masked_pos.numel() * sizeof(int64_t));
  std::memcpy(isNext.data_ptr(), _isNext.data(), isNext.numel() * sizeof(int64_t));

  input_ids_ = input_ids;
  segment_ids_ = segment_ids;
  masked_tokens_ = masked_tokens;
  masked_pos_ = masked_pos;
  isNext_ = isNext;

}

int64_t NLP::getClassNumber() {
  return words.size();
}

BERTExample NLP::get(size_t index) {
  return { {input_ids_[index], segment_ids_[index], masked_tokens_[index], masked_pos_[index]}, isNext_[index]};
}

optional<size_t> NLP::size() const {
  return input_ids_.size(0);
}

const std::string& NLP::index_to_string(int64_t index) {
  return index_word[index];
}


} // namespace datasets
} // namespace data
} // namespace torch
