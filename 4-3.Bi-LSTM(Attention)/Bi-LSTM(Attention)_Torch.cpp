
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


// Bi-LSTM(Attention) Parameters
const int64_t embedding_dim = 2;
const int64_t n_hidden = 5; // number of hidden units in one cell
const int64_t num_classes = 2;  // 0 or 1

std::shared_ptr<torch::nn::LSTMImpl> make_lstm(int64_t input_size, int64_t hidden_size) {
  auto options = torch::nn::LSTMOptions(input_size, hidden_size);
  options.bidirectional(true);
  return std::make_shared<torch::nn::LSTMImpl>(options);
}

std::shared_ptr<torch::nn::LinearImpl> make_linear(int64_t input_size, int64_t hidden_size) {
  auto options = torch::nn::LinearOptions(input_size, hidden_size);
  return std::make_shared<torch::nn::LinearImpl>(options);
}


struct Net : torch::nn::Module {

  Net(int64_t vocab_size, torch::Device device)
    : device(device),
      embedding(torch::nn::EmbeddingOptions(vocab_size, embedding_dim)),
      lstm(make_lstm(embedding_dim, n_hidden)),
      out(make_linear(n_hidden * 2, num_classes))
  {
    register_module("embedding", embedding);
    register_module("lstm", lstm);
    register_module("out", out);
  }

  // lstm_output : [batch_size, n_step, n_hidden * num_directions(=2)], F matrix
  std::pair<torch::Tensor,torch::Tensor> attention_net(torch::Tensor lstm_output, torch::Tensor final_state) {

    auto hidden = final_state.view({-1, n_hidden * 2, 1}); // hidden : [batch_size, n_hidden * num_directions(=2), 1(=n_layer)]
    auto attn_weights = torch::bmm(lstm_output, hidden).squeeze(2);  // attn_weights : [batch_size, n_step]
    auto soft_attn_weights = at::softmax(attn_weights, 1);
    // [batch_size, n_hidden * num_directions(=2), n_step] * [batch_size, n_step, 1] = [batch_size, n_hidden * num_directions(=2), 1]
    auto context = torch::bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2).to(device);
    return std::make_pair(context, soft_attn_weights); // context : [batch_size, n_hidden * num_directions(=2)]
  }

  torch::Tensor forward(torch::Tensor X) {

    auto input = embedding(X); // input : [batch_size, len_seq, embedding_dim]
    input = input.permute({1, 0, 2}); // input : [len_seq, batch_size, embedding_dim]

    auto hidden_state = torch::zeros({1*2, X.sizes().size(), n_hidden}).to(device); // [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
    auto cell_state = torch::zeros({1*2, X.sizes().size(), n_hidden}).to(device);  // [num_layers(=1) * num_directions(=2), batch_size, n_hidden]

    // final_hidden_state, final_cell_state : [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
    auto lstmOutput = lstm->forward(input, torch::cat({hidden_state, cell_state}, 0));
    auto output = lstmOutput.output;
    auto final_hidden_state = lstmOutput.state[0];
    auto final_cell_state = lstmOutput.state[1];
    output = output.permute({1, 0, 2}); // output : [batch_size, len_seq, n_hidden]
    auto attn = attention_net(output, final_hidden_state);
    auto attn_output = attn.first;
    auto attention = attn.second;
    attn_output = out->forward(attn_output);
    attn_output = torch::log_softmax(attn_output, /*dim=*/1);
    return attn_output; // attention; // model : [batch_size, num_classes], attention : [batch_size, n_step]
  }

  torch::Device device;
  torch::nn::Embedding embedding;
  std::shared_ptr<torch::nn::LSTMImpl> lstm;
  std::shared_ptr<torch::nn::LinearImpl> out;


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
  Net model(nClass, device);
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


