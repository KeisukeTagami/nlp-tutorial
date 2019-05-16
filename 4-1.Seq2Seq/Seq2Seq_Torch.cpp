
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

const int64_t n_hidden = 128;

std::shared_ptr<torch::nn::RNNImpl> make_rnn(int64_t input_size, int64_t hidden_size) {
  auto options = torch::nn::RNNOptions(input_size, hidden_size);
  options.dropout(0.5);
  return std::make_shared<torch::nn::RNNImpl>(options);
}

std::shared_ptr<torch::nn::LinearImpl> make_linear(int64_t input_size, int64_t hidden_size) {
  auto options = torch::nn::LinearOptions(input_size, hidden_size);
  return std::make_shared<torch::nn::LinearImpl>(options);
}


struct Net : torch::nn::Module {
  Net(int64_t n_class)
    : enc_cell(make_rnn(n_class, n_hidden)),
      dec_cell(make_rnn(n_class, n_hidden)),
      fc(make_linear(n_hidden, n_class))
  {
    register_module("enc_cell", enc_cell);
    register_module("dec_cell", dec_cell);
    register_module("fc", fc);
  }

  torch::Tensor forward(torch::Tensor enc_input, torch::Tensor enc_hidden, torch::Tensor dec_input) {
    enc_input = enc_input.transpose(0, 1); // enc_input : [n_step, batch_size, n_class]
    dec_input = dec_input.transpose(0, 1); // enc_input : [n_step, batch_size, n_class]
    auto enc_output = enc_cell->forward(enc_input, enc_hidden);
    auto dec_output = dec_cell->forward(dec_input, enc_output.state);
    auto output = fc->forward(dec_output.output);
    output = torch::log_softmax(output, /*dim=*/2);
    return output;
  }

  std::shared_ptr<torch::nn::RNNImpl> enc_cell;
  std::shared_ptr<torch::nn::RNNImpl> dec_cell;
  std::shared_ptr<torch::nn::LinearImpl> fc;
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
  torch::Device device(device_type);

  std::vector<std::pair<std::string, std::string>>
    seq_data {{"man", "women"},
              {"black", "white"},
              {"king", "queen"},
              {"girl", "boy"},
              {"up", "down"},
              {"high", "low"}};

  const int64_t batch_size = static_cast<int64_t>(seq_data.size());

  auto dataset = torch::data::datasets::NLP(seq_data);
  auto train_dataset = dataset.map(torch::data::transforms::Stack<>());
  auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(train_dataset), batch_size);

 int64_t nClass = dataset.getClassNumber();
  Net model(nClass);

  model.to(device);
  torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(0.001));
  model.train();

  for (size_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {

    float loss_value;
    torch::Tensor loss;

    for (auto& batch : *data_loader) {
      auto input = batch.data.to(device);
      auto targets = batch.target.to(device);
      auto hidden = torch::Tensor(torch::zeros({1, batch_size, n_hidden})).to(device);
      targets = targets.transpose(0, 1);
      auto i = torch::one_hot(input, dataset.getClassNumber()).to(torch::kFloat);
      auto o = torch::one_hot(targets[0].view({batch_size, -1}), dataset.getClassNumber()).to(torch::kFloat);
      auto t = targets[1].view({batch_size, -1});

      optimizer.zero_grad();
      auto output = model.forward(i, hidden, o);
      output = output.transpose(0, 1); // [batch_size, max_len+1(=6), num_directions(=1) * n_hidden]

      for( int i = 0 ; i < batch_size; i++ )
        if( loss.defined() )
          loss += torch::nll_loss(output[i], t[i]);
        else
          loss = torch::nll_loss(output[i], t[i]);
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
    auto hidden = torch::Tensor(torch::zeros({1, batch_size, n_hidden})).to(device);
    targets = targets.transpose(0, 1);
    auto i = torch::one_hot(input, dataset.getClassNumber()).to(torch::kFloat);
    auto o = torch::one_hot(targets[0].view({batch_size, -1}), dataset.getClassNumber()).to(torch::kFloat);
    auto t = targets[1].view({batch_size, -1});
    auto predict = model.forward(i, hidden, o);
    predict = predict.transpose(0, 1); // [batch_size, max_len+1(=6), num_directions(=1) * n_hidden]

    input = i.argmax(2).cpu();
    targets = t.cpu();
    predict = predict.argmax(2).cpu();

    auto input_accessor = input.accessor<int64_t,2>();
    auto targets_accessor = targets.accessor<int64_t,2>();
    auto predict_accessor = predict.accessor<int64_t,2>();

    std::cout << std::endl;
    for(int i = 0; i < input_accessor.size(0); i++) {
      for(int j = 0; j < input_accessor.size(1); j++) {
        std::cout << dataset.index_to_string(input_accessor[i][j]);
      }
      std::cout << " ";
      for(int j = 0; j < predict_accessor.size(1); j++) {
        std::cout << dataset.index_to_string(predict_accessor[i][j]);
      }

      std::cout << " [";
      for(int j = 0; j < targets_accessor.size(1); j++) {
        std::cout << dataset.index_to_string(targets_accessor[i][j]);
      }
      std::cout << "]";

      std::cout << std::endl;

    }
  }
}


