/********************************************
 * $ mkdir build
 * $ cd build
 * $ cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
 * $ make
 **********************************************/

#include <torch/torch.h>

#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "NLPDataset.h"

// The batch size for training.
const int64_t kTrainBatchSize = 3;

// The batch size for testing.
const int64_t kTestBatchSize = 1;

// The number of epochs to train.
const int64_t kNumberOfEpochs = 500;

// After how many batches to log a new update with the loss value.
const int64_t kLogInterval = 1000;

const int64_t nStep = 2;
const int64_t nHidden = 2;
const int64_t m = 2;

struct Net : torch::nn::Module {
  Net(int64_t nClass)
    : C(torch::nn::EmbeddingOptions(nClass, m)),
      H(torch::rand({nStep * m, nHidden})),
      W(torch::rand({nStep * m, nClass})),
      d(torch::rand({nHidden})),
      U(torch::rand({nHidden, nClass})),
      b(torch::rand({nClass}))
  {
    register_module("C", C);
    register_parameter("H", H);
    register_parameter("W", W);
    register_parameter("d", d);
    register_parameter("U", U);
    register_parameter("b", b);
  }

  torch::Tensor forward(torch::Tensor x) {
    x = C(x);
    x = x.view({-1, nStep * m});
    torch::Tensor tanh = at::tanh(d + at::matmul(x, H));
    torch::Tensor output = b + at::matmul(x, W) + at::matmul(tanh, U);
    return torch::log_softmax(output, /*dim=*/1);
  }

  torch::nn::Embedding C;
  torch::Tensor H;
  torch::Tensor W;
  torch::Tensor d;
  torch::Tensor U;
  torch::Tensor b;
};

auto main() -> int {
  torch::manual_seed(1);

  torch::DeviceType device_type;
  if (torch::cuda::is_available()) {
    std::cout << "CUDA available! Training on GPU." << std::endl;
    device_type = torch::kCUDA;
  } else {
    std::cout << "Training on CPU." << std::endl;
    device_type = torch::kCPU;
  }
  // device_type = torch::kCPU;
  torch::Device device(device_type);

  std::vector<std::string>   sentences{ "i like dog", "i love coffee", "i hate milk"};
  auto dataset = torch::data::datasets::NLP(sentences);
  auto train_dataset = dataset.map(torch::data::transforms::Stack<>());
  auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(train_dataset), kTrainBatchSize);


  int64_t nClass = dataset.getClassNumber();
  Net model(nClass);
  model.to(device);

  torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(0.001));

  model.train();
  for (size_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {

    float loss_value;

    for (auto& batch : *data_loader) {
      auto data = batch.data.to(device);
      auto targets = batch.target.to(device);
      optimizer.zero_grad();
      auto output = model.forward(data);
      auto loss = torch::nll_loss(output, targets);
      AT_ASSERT(!std::isnan(loss.template item<float>()));
      loss.backward();
      optimizer.step();

      loss_value = loss.template item<float>();
    }

    if (epoch % kLogInterval == 0) {
      std::cout << std::endl;
      std::cout << "\rTrain Epoch: " << epoch
                << "[" << std::setfill(' ') << std::setw(5) << epoch
                << "/" << std::setfill(' ') << std::setw(5) << kNumberOfEpochs
                << "]"
                << " Loss: " <<  loss_value;
    }
  }

  model.eval();
  auto input = dataset.input().to(device);
  auto targets = dataset.targets();
  auto predict = model.forward(input).argmax(1);
  input = input.cpu();
  predict = predict.cpu();

  auto input_accessor = input.accessor<int64_t,2>();
  auto targets_accessor = targets.accessor<int64_t,1>();
  auto predict_accessor = predict.accessor<int64_t,1>();

  std::cout << std::endl;
  for(int i = 0; i < input_accessor.size(0); i++) {
    for(int j = 0; j < input_accessor.size(1); j++) {
      std::cout << dataset.index_to_string(input_accessor[i][j]) << " ";
    }
    std::cout << dataset.index_to_string(predict_accessor[i]);
    std::cout << " [" << dataset.index_to_string(targets_accessor[i]) << "]";
    std::cout << std::endl;
  }
}


