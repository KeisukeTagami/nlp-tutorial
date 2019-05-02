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

// The number of epochs to train.
const int64_t kNumberOfEpochs = 5000;

// After how many batches to log a new update with the loss value.
const int64_t kLogInterval = 1000;

const int64_t n_step = 3;
const int64_t n_hidden = 128;

std::shared_ptr<torch::nn::LSTMImpl> make_lstm(int64_t input_size, int64_t hidden_size) {
  auto options = torch::nn::LSTMOptions(input_size, hidden_size);
  return std::make_shared<torch::nn::LSTMImpl>(options);
}

struct Net : torch::nn::Module {
  Net(int64_t n_class)
    : lstm(make_lstm(n_class, n_hidden)),
      W(torch::rand({n_hidden, n_class})),
      b(torch::rand({n_class}))
  {
    register_module("lstm", lstm);
    register_parameter("W", W);
    register_parameter("b", b);
  }

  torch::Tensor forward(torch::Tensor X) {
    X = X.transpose(0, 1); // X : [n_step, batch_size, n_class]

    auto hidden_state = torch::Tensor(torch::zeros({1, 1, X.sizes().vec().at(1), n_hidden})); // [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
    auto cell_state = torch::Tensor(torch::zeros({1, 1, X.sizes().vec().at(1), n_hidden})); // [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
    auto state = torch::cat({hidden_state, cell_state}, 0);

    auto outputs = lstm->forward(X, state).output;
    outputs = outputs[-1]; //[batch_size, num_directions(=1) * n_hidden]
    outputs = torch::mm(outputs, W) + b; // model : [batch_size, n_class]
    outputs = torch::log_softmax(outputs, /*dim=*/1);
    return outputs;
  }

  std::shared_ptr<torch::nn::LSTMImpl> lstm;
  torch::Tensor W;
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

  std::vector<std::string> words{ "make", "need", "coal", "word", "love", "hate", "live", "home", "hash", "star"};

  const int64_t batch_size = static_cast<int64_t>(words.size());

  auto dataset = torch::data::datasets::NLP(words);
  auto train_dataset = dataset.map(torch::data::transforms::Stack<>());
  auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(train_dataset), batch_size);

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

  for (auto& batch : *data_loader) {
    auto input = batch.data.to(device);
    auto targets = batch.target.to(device);
    auto predict = model.forward(input);
    input = input.argmax(2).cpu();
    predict = predict.argmax(1).cpu();

    auto input_accessor = input.accessor<int64_t,2>();
    auto targets_accessor = targets.accessor<int64_t,1>();
    auto predict_accessor = predict.accessor<int64_t,1>();

    std::cout << std::endl;
    for(int i = 0; i < input_accessor.size(0); i++) {
      for(int j = 0; j < input_accessor.size(1); j++) {
        std::cout << dataset.index_to_string(input_accessor[i][j]);
      }
      std::cout << " ";
      std::cout << dataset.index_to_string(predict_accessor[i]);
      std::cout << " [" << dataset.index_to_string(targets_accessor[i]) << "]";
      std::cout << std::endl;
    }
  }
}


