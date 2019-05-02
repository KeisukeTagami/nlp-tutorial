
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
const int64_t kTrainBatchSize = 6;

// The number of epochs to train.
const int64_t kNumberOfEpochs = 5000;

// After how many batches to log a new update with the loss value.
const int64_t kLogInterval = 1000;


// Text-CNN Parameter
const int64_t embedding_size = 2; // n-gram
const int64_t sequence_length = 3;
const int64_t num_classes = 2; //# 0 or 1
const std::vector<int64_t> filter_sizes = {2, 2, 2}; // n-gram window
const int64_t num_filters = 3;

torch::Tensor getElementWith(torch::Tensor& index, torch::Tensor& target) {
  auto sizes = index.sizes().vec();
  int64_t element_size = 1;
  for( auto s : sizes ) {
    element_size *= s;
  }
  sizes.push_back(-1);
  index = index.view({-1});
  std::vector<torch::Tensor> vec;
  for( int64_t i = 0 ; i < element_size ; i++ ) {
    auto t = target[index[i]];
    vec.push_back(t);
  }
  return torch::cat(vec, -1).view(sizes);
}


std::shared_ptr<torch::nn::Conv2dImpl> conv2d(int64_t input_channels, int64_t output_channels, torch::ExpandingArray<2> kernel_size) {
  auto options = torch::nn::Conv2dOptions(input_channels, output_channels, kernel_size).stride(1).with_bias(true);
  return std::make_shared<torch::nn::Conv2dImpl>(options);
}

struct Net : torch::nn::Module {

  Net(int64_t vocab_size)
    : num_filters_total(num_filters * filter_sizes.size()),
      W(torch::empty({vocab_size, embedding_size}).uniform_(-1, 1)),
      Weight(torch::empty({num_filters_total, num_classes}).uniform_(-1, 1)),
      Bias(0.1 * torch::ones({num_classes}))
  {
    int idx = 0;
    for( auto filter_size : filter_sizes) {
      auto C = conv2d(1, num_filters, {filter_size, embedding_size});
      register_module("C" + std::to_string(idx++) , C);
      Cs.push_back(C);
    }

    register_parameter("W", W);
    register_parameter("Weight", Weight);
    register_parameter("Bias", Bias);
  }

  torch::Tensor forward(torch::Tensor X) {
    auto embedded_chars = getElementWith(X, W); // [batch_size, sequence_length, embedding_size]
    embedded_chars = embedded_chars.unsqueeze(1); // add channel(=1) [batch, channel(=1), sequence_length, embedding_size]

    // embedded_chars.print();
    std::vector<torch::Tensor> pooled_outputs;
    for( auto C : Cs ) {
      auto kernel_size = C->options.kernel_size();
      auto filter_size = kernel_size->at(0);
      // conv : [input_channel(=1), output_channel(=3), (filter_height, filter_width), bias_option]
      auto conv = C->forward(embedded_chars);
      auto h = at::relu(conv);
      // mp : ((filter_height, filter_width))
      auto mp = torch::max_pool2d(h,{sequence_length - filter_size + 1, 1});
      // pooled : [batch_size(=6), output_height(=1), output_width(=1), output_channel(=3)]
      auto pooled = mp.permute({0, 3, 2, 1});
      pooled_outputs.push_back(pooled);
    }
    auto h_pool = torch::cat(pooled_outputs, filter_sizes.size()); // [batch_size(=6), output_height(=1), output_width(=1), output_channel(=3) * 3];
    auto h_pool_flat = torch::reshape(h_pool, {-1, num_filters_total}); // [batch_size(=6), output_height * output_width * (output_channel * 3)];

    auto output = torch::mm(h_pool_flat, Weight) + Bias; // [batch_size, num_classes]
    return torch::log_softmax(output, /*dim=*/1);
  }

  int64_t num_filters_total;
  std::vector<std::shared_ptr<torch::nn::Conv2dImpl>> Cs;
  torch::Tensor W;
  torch::Tensor Weight;
  torch::Tensor Bias;
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

  std::vector<std::string> sentences {"i love you", "he loves me", "she likes baseball", "i hate you", "sorry for that", "this is awful"};
  std::vector<int64_t> labels {1, 1, 1, 0, 0, 0}; // 1 is good, 0 is not good.

  auto dataset = torch::data::datasets::NLP(sentences, labels);

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
                << " Loss: " << loss_value;
    }


  }

  model.eval();

  std::vector<int64_t> test_sentences;
  for( auto word : std::vector<std::string>({ "sorry", "hate", "you"} )) {
    test_sentences.push_back(dataset.string_to_index(word));
  }
  auto test_text = torch::empty({1, 3}, torch::kInt64);
  std::memcpy(test_text.data_ptr(), test_sentences.data(), test_text.numel() * sizeof(int64_t));

  auto input = test_text.to(device);
  auto predict = model.forward(input).argmax(1);
  input = input.cpu();
  predict = predict.cpu();

  auto input_accessor = input.accessor<int64_t,2>();
  auto predict_accessor = predict.accessor<int64_t,1>();

  std::cout << std::endl;
  for(int i = 0; i < input_accessor.size(0); i++) {
    for(int j = 0; j < input_accessor.size(1); j++) {
      std::cout << dataset.index_to_string(input_accessor[i][j]) << " ";
    }
    std::cout << "is " << (predict_accessor[i] == 0 ? "Bad Mean..." : "Good Mean!!");
    std::cout << std::endl;
  }

}


