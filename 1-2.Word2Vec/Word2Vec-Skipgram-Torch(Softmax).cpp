
/********************************************
 *
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
const int64_t kTrainBatchSize = 20;

// The batch size for testing.
const int64_t kTestBatchSize = 1;

// The number of epochs to train.
const int64_t kNumberOfEpochs = 500;

// After how many batches to log a new update with the loss value.
const int64_t kLogInterval = 1000;

const int64_t nEmbedding = 2;
const int64_t m = 2;

struct Net : torch::nn::Module {
  Net(int64_t nClass)
    : C(torch::nn::EmbeddingOptions(kTrainBatchSize, nClass)),
      W(torch::rand({nClass, nEmbedding}) * -2 + 1 ),
      WT(torch::rand({nEmbedding, nClass}) * -2 + 1  )
  {
    register_module("C", C);
    register_parameter("W", W);
    register_parameter("WT", WT);
  }

  torch::Tensor forward(torch::Tensor x) {
    x = C(x);
    torch::Tensor hidden_layer = at::matmul(x, W);
    torch::Tensor output = at::matmul(hidden_layer, WT);
    return torch::log_softmax(output, /*dim=*/1);
  }

  torch::nn::Embedding C;
  torch::Tensor W;
  torch::Tensor WT;
};

template <typename DataLoader>
void train(
    int32_t epoch,
    Net& model,
    torch::Device device,
    DataLoader& data_loader,
    torch::optim::Optimizer& optimizer,
    size_t dataset_size) {

  model.train();
  size_t batch_idx = 0;
  for (auto& batch : data_loader) {
    auto data = batch.data.to(device);
    auto targets = batch.target.to(device);
    optimizer.zero_grad();
    auto output = model.forward(data);
    auto loss = torch::nll_loss(output, targets);
    AT_ASSERT(!std::isnan(loss.template item<float>()));
    loss.backward();
    optimizer.step();

    if (batch_idx++ % kLogInterval == 0) {
      std::cout << std::endl;
      std::cout << "\rTrain Epoch: " << epoch
                << "[" << std::setfill(' ') << std::setw(5) << batch_idx * batch.data.size(0)
                << "/" << std::setfill(' ') << std::setw(5) << dataset_size
                << "]"
                << " Loss: " <<  loss.template item<float>();
    }

  }
}

template <typename DataLoader>
void test(
    Net& model,
    torch::Device device,
    DataLoader& data_loader,
    size_t dataset_size) {
  torch::NoGradGuard no_grad;
  model.eval();
  double test_loss = 0;
  int32_t correct = 0;
  for (const auto& batch : data_loader) {
    auto data = batch.data.to(device);
    auto targets = batch.target.to(device);
    auto output = model.forward(data);
    test_loss += torch::nll_loss(output, targets,
                                 /*weight=*/{},
                                 Reduction::Sum).template item<float>();
    auto pred = output.argmax(1);
    correct += pred.eq(targets).sum().template item<int64_t>();
  }

  test_loss /= dataset_size;
  std::cout << std::endl;
  std::cout << "Test set: "
            << "Average loss: " << std::fixed << std::setprecision(4) << test_loss
            << " | Accuracy: "  << std::fixed << std::setprecision(3) << static_cast<double>(correct) / dataset_size;
}

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

  std::vector<std::string>   sentences{ "i like dog", "i like cat", "i like animal",
                                        "dog cat animal", "apple cat dog like", "dog fish milk like",
                                        "dog cat eyes like", "i like apple", "apple i hate",
                                        "apple i movie book music like", "cat dog hate", "cat dog like"};
  auto dataset = torch::data::datasets::NLP(sentences);

  auto train_dataset = dataset.map(torch::data::transforms::Stack<>());
  const size_t train_dataset_size = train_dataset.size().value();
  auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(train_dataset), kTrainBatchSize);

  auto test_dataset = dataset.map(torch::data::transforms::Stack<>());
  const size_t test_dataset_size = test_dataset.size().value();
  auto test_loader = torch::data::make_data_loader(std::move(test_dataset), kTestBatchSize);

  int64_t nClass = dataset.getClassNumber();
  Net * model = new Net(nClass);
  model->to(device);

  torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(0.001));

  for (size_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
    train(epoch, *model, device, *train_loader, optimizer, train_dataset_size);
    test(*model, device, *test_loader, test_dataset_size);
  }

  model->eval();

  std::vector<torch::Tensor> parameters = model->parameters();
  std::cout << parameters.size() << std::endl;
  if( 3 == parameters.size() ) {
    torch::Tensor W = parameters[1].cpu();
    torch::Tensor WT = parameters[2].cpu();

    auto W_accessor = W.accessor<float,2>();
    auto WT_accessor = WT.accessor<float,2>();

    for( auto idx = 0 ; idx < dataset.getClassNumber(); idx++ ) {
      std::cout << dataset.index_to_string(idx)
                << ": "
                << W_accessor[idx][0]
                << ", "
                << WT_accessor[idx][1]
                << std::endl;
    }
  }

}


