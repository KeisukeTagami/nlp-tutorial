#pragma once

#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/types.h>
#include <torch/data/transforms/stack.h>

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <cstddef>
#include <string>

namespace torch {
namespace data {

/// The NLP dataset.
  typedef Example<std::tuple<Tensor, Tensor, Tensor, Tensor>, Tensor> BERTExample;

namespace transforms {

template <>
struct Stack<BERTExample>
  : public Collation<BERTExample> {
  BERTExample apply_batch(std::vector<BERTExample> examples) override {
    std::vector<torch::Tensor> input_ids, segment_ids, masked_tokens, masked_pos, isNext;
    input_ids.reserve(examples.size());
    segment_ids.reserve(examples.size());
    masked_tokens.reserve(examples.size());
    masked_pos.reserve(examples.size());
    isNext.reserve(examples.size());
    for (auto& example : examples) {
      input_ids.push_back(std::move(std::get<0>(example.data)));
      segment_ids.push_back(std::move(std::get<1>(example.data)));
      masked_tokens.push_back(std::move(std::get<2>(example.data)));
      masked_pos.push_back(std::move(std::get<3>(example.data)));
      isNext.push_back(std::move(example.target));
    }
    return {{torch::stack(input_ids), torch::stack(segment_ids), torch::stack(masked_tokens), torch::stack(masked_pos)}, torch::stack(isNext)};
  }
};

}

namespace datasets {

class TORCH_API NLP : public Dataset<NLP, BERTExample> {

 public:

  explicit NLP(std::vector<std::string> seq_data);

  int64_t getClassNumber();

  /// Returns the `Example` at the given `index`.
  BERTExample get(size_t index) override;

  /// Returns the size of the dataset.
  optional<size_t> size() const override;

  const std::string& index_to_string(int64_t);

 private:
  Tensor input_ids_, segment_ids_, masked_tokens_, masked_pos_, isNext_;

  std::set<std::string>          words;
  std::map<std::string, int64_t> word_index;
  std::map<int64_t, std::string> index_word;
};
} // namespace datasets
} // namespace data
} // namespace torch

