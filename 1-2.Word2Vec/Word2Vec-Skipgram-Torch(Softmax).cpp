
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
const int64_t kTrainBatchSize = 20;

// The number of epochs to train.
const int64_t kNumberOfEpochs = 5000;

// After how many batches to log a new update with the loss value.
const int64_t kLogInterval = 1000;

const int64_t nEmbedding = 2;

struct Net : torch::nn::Module {
  Net(int64_t nClass)
    : W(torch::rand({nClass, nEmbedding}) * -2 + 1 ),
      WT(torch::rand({nEmbedding, nClass}) * -2 + 1 )
  {
    register_parameter("W", W);
    register_parameter("WT", WT);
  }

  torch::Tensor forward(torch::Tensor x) {
    torch::Tensor hidden_layer = at::matmul(x, W);
    torch::Tensor output = at::matmul(hidden_layer, WT);
    return torch::log_softmax(output, 1);
  }

  torch::Tensor W;
  torch::Tensor WT;
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

  std::vector<std::string>   sentences{ "i like dog", "i like cat", "i like animal",
                                        "dog cat animal", "apple cat dog like", "dog fish milk like",
                                        "dog cat eyes like", "i like apple", "apple i hate",
                                        "apple i movie book music like", "cat dog hate", "cat dog like"};
  auto dataset = torch::data::datasets::NLP(sentences);

  auto train_dataset = dataset.map(torch::data::transforms::Stack<>());
  auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(train_dataset), kTrainBatchSize);

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
  std::vector<torch::Tensor> parameters = model.parameters();
  std::cout << parameters.size() << std::endl;
  if( 2 == parameters.size() ) {
    torch::Tensor W = parameters[0].cpu();
    auto W_accessor = W.accessor<float,2>();

    for( auto idx = 0 ; idx < dataset.getClassNumber(); idx++ ) {
      std::cout << dataset.index_to_string(idx)
                << ", " << W_accessor[idx][0]
                << ", " << W_accessor[idx][1]
                << std::endl;
    }
  }
}


