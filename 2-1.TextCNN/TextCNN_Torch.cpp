
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
const int64_t kTrainBatchSize = 3;

// The batch size for testing.
const int64_t kTestBatchSize = 1;

// The number of epochs to train.
const int64_t kNumberOfEpochs = 500;

// After how many batches to log a new update with the loss value.
const int64_t kLogInterval = 1000;


// Text-CNN Parameter
const int64_t embedding_size = 2; // n-gram
const int64_t sequence_length = 3;
const int64_t num_classes = 2; //# 0 or 1
const std::vector<int64_t> filter_sizes = {2, 2, 2}; // n-gram window
const int64_t num_filters = 3;

struct Net : torch::nn::Module {

  Net(int64_t vocab_size)
    : num_filters_total(num_filters * filter_sizes.size()),
      W(torch::empty({vocab_size, embedding_size}).uniform_(-1, 1)),
      Weight(torch::empty({num_filters_total, num_classes}).uniform_(-1, 1)),
      Bias(0.1 * torch::ones({num_classes}))
  {
    register_parameter("W", W);
    register_parameter("Weight", Weight);
    register_parameter("Bias", Bias);
  }

  torch::Tensor forward(torch::Tensor X) {

    auto embedded_chars = W[X]; // [batch_size, sequence_length, sequence_length]
    embedded_chars = embedded_chars.unsqueeze(1); // add channel(=1) [batch, channel(=1), sequence_length, embedding_size]

    std::vector<torch::Tensor> pooled_outputs;
    for( auto filter_size : filter_sizes) {
      // conv : [input_channel(=1), output_channel(=3), (filter_height, filter_width), bias_option]
      auto conv = torch::nn::Conv2d(1, num_filters, {filter_size, embedding_size}, true)(embedded_chars);
      auto h = at::relu(conv);
      // mp : ((filter_height, filter_width))
      auto mp = torch::nn::MaxPool2d({sequence_length - filter_size + 1, 1});
      // pooled : [batch_size(=6), output_height(=1), output_width(=1), output_channel(=3)]
      auto pooled = mp(h).permute(0, 3, 2, 1);
      pooled_outputs.push_back(pooled);

      auto h_pool = torch.cat(pooled_outputs, filter_sizes.size()); // [batch_size(=6), output_height(=1), output_width(=1), output_channel(=3) * 3];
      auto h_pool_flat = torch.reshape(h_pool, {-1, num_filters_total}); // [batch_size(=6), output_height * output_width * (output_channel * 3)];

      return torch::mm(h_pool_flat, self.Weight) + Bias; // [batch_size, num_classes]
    }

    return x;
  }

  int64_t num_filters_total;
  torch::Tensor W;
  torch::Tensor Weight;
  torch::Tensor Bias;

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

  std::vector<std::string>   sentences{ "i like dog", "i love coffee", "i hate milk"};
  auto dataset = torch::data::datasets::NLP(sentences);

  auto train_dataset = dataset.map(torch::data::transforms::Stack<>());
  const size_t train_dataset_size = train_dataset.size().value();
  auto train_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(train_dataset), kTrainBatchSize);

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
  auto input = dataset.input().to(device);
  auto targets = dataset.targets();
  auto predict = model->forward(input).argmax(1);
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


