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
typedef Example<std::pair<Tensor, Tensor>, Tensor> RNNExample;

namespace transforms {

template <>
struct Stack<RNNExample>
  : public Collation<RNNExample> {
  RNNExample apply_batch(std::vector<RNNExample> examples) override {
    std::vector<torch::Tensor> firsts, seconds, targets;
    firsts.reserve(examples.size());
    seconds.reserve(examples.size());
    targets.reserve(examples.size());
    for (auto& example : examples) {
      firsts.push_back(std::move(example.data.first));
      seconds.push_back(std::move(example.data.second));
      targets.push_back(std::move(example.target));
    }
    return {{torch::stack(firsts), torch::stack(seconds)}, torch::stack(targets)};
  }
};

}

namespace datasets {

class TORCH_API NLP : public Dataset<NLP, RNNExample> {

 public:

  /// The mode in which the dataset is loaded.
  enum class Mode { kTrain, kTest };

  /// Loads the NLP dataset from the `root` path.
  ///
  /// The supplied `root` path should contain the *content* of the unzipped
  /// NLP dataset, available from http://yann.lecun.com/exdb/mnist.
  explicit NLP(std::vector<std::pair<std::string, std::string>> seq_data);

  bool setMode(Mode mode);
  int64_t getClassNumber();

  /// Returns the `Example` at the given `index`.
  RNNExample get(size_t index) override;

  /// Returns the size of the dataset.
  optional<size_t> size() const override;

  /// Returns all images stacked into a single tensor.
  const Tensor& input() const;

  /// Returns all images stacked into a single tensor.
  const Tensor& output() const;

  /// Returns all targets stacked into a single tensor.
  const Tensor& targets() const;

  const std::string& index_to_string(int64_t);

  Tensor strings_to_tensor(std::vector<std::string> sentences);

 private:
  Tensor input_, output_, targets_;

  std::set<std::string>          words;
  std::map<std::string, int64_t> word_index;
  std::map<int64_t, std::string> index_word;
};
} // namespace datasets
} // namespace data
} // namespace torch
