#pragma once

#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/types.h>

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <cstddef>
#include <string>

namespace torch {
namespace data {
namespace datasets {
/// The NLP dataset.
class TORCH_API NLP : public Dataset<NLP> {
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
  Example<> get(size_t index) override;

  /// Returns the size of the dataset.
  optional<size_t> size() const override;

  /// Returns all images stacked into a single tensor.
  const Tensor& input() const;

  /// Returns all targets stacked into a single tensor.
  const Tensor& targets() const;

  const std::string& index_to_string(int64_t);

 private:
  Tensor input_, targets_;

  std::set<std::string>          words;
  std::map<std::string, int64_t> word_index;
  std::map<int64_t, std::string> index_word;
};
} // namespace datasets
} // namespace data
} // namespace torch
