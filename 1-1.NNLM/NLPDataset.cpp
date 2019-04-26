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
constexpr uint32_t kTrainSize = 60000;
constexpr uint32_t kTestSize = 10000;
constexpr uint32_t kImageRows = 1;
constexpr uint32_t kImageColumns = 2;
constexpr uint32_t kTarget  = 7;

Tensor read_images(bool train) {
  const auto count = train ? kTrainSize : kTestSize;
  auto tensor = torch::rand({count, 1, kImageRows, kImageColumns});
  return tensor.to(torch::kInt64);
}

Tensor read_targets(bool train) {
  const auto count = train ? kTrainSize : kTestSize;
  auto tensor = torch::rand({count, kTarget});
  return tensor.to(torch::kFloat32);
}
} // namespace

NLP::NLP(Mode mode)
    : images_(read_images(mode == Mode::kTrain)),
      targets_(read_targets(mode == Mode::kTrain)) {}

Example<> NLP::get(size_t index) {
  return {images_[index], targets_[index]};
}

optional<size_t> NLP::size() const {
  return images_.size(0);
}

bool NLP::is_train() const noexcept {
  return images_.size(0) == kTrainSize;
}

const Tensor& NLP::images() const {
  return images_;
}

const Tensor& NLP::targets() const {
  return targets_;
}

} // namespace datasets
} // namespace data
} // namespace torch
