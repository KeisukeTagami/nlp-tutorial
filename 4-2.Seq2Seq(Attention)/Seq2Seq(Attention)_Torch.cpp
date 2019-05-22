

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
const int64_t kNumberOfEpochs = 100;

// After how many batches to log a new update with the loss value.
const int64_t kLogInterval = 10;

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
  Net(int64_t n_class, torch::Device device)
    : n_class(n_class),
      device(device),
      enc_cell(make_rnn(n_class, n_hidden)),
      dec_cell(make_rnn(n_class, n_hidden)),
      attn(make_linear(n_hidden, n_hidden)),
      out(make_linear(n_hidden * 2 , n_class))
  {
    register_module("enc_cell", enc_cell);
    register_module("dec_cell", dec_cell);
    register_module("attn", attn);
    register_module("out", out);
  }

  torch::Tensor forward(torch::Tensor enc_inputs, torch::Tensor hidden, torch::Tensor dec_inputs) {
    enc_inputs = enc_inputs.transpose(0, 1); // enc_input : [n_step, batch_size, n_class]
    dec_inputs = dec_inputs.transpose(0, 1); // enc_input : [n_step, batch_size, n_class]
    auto outputs = enc_cell->forward(enc_inputs, hidden);
    auto enc_outputs = outputs.output;
    hidden = outputs.state;

    std::vector<torch::Tensor> trained_attn;
    auto n_step = dec_inputs.sizes().vec().at(0);
    auto batch_size = dec_inputs.sizes().vec().at(1);
    auto model = torch::empty({n_step, batch_size, n_class}).to(device);

    for( int i=0 ; i < n_step; i++ ) { //  each time step
      // dec_output : [n_step(=1), batch_size, num_directions(=1) * n_hidden]
      // hidden : [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
      outputs = dec_cell->forward(dec_inputs[i].unsqueeze(0), hidden);
      auto dec_output = outputs.output;
      hidden = outputs.state;

      auto attn_weights = get_att_weight(dec_output, enc_outputs); // attn_weights : [batch_size, 1, n_step]
      trained_attn.push_back(attn_weights.squeeze());

      // matrix-matrix product of matrices [batch_size, 1, n_step] x [batch_size, n_step, n_hidden] = [1,1,n_hidden]
      auto context = attn_weights.bmm(enc_outputs.transpose(0, 1));
      dec_output = dec_output.squeeze(0); // dec_output : [batch_size, num_directions(=1) * n_hidden]
      context = context.squeeze(1);  // [1, num_directions(=1) * n_hidden]
      model[i] = out->forward(torch::cat({dec_output, context}, 1));
    }

    // make model shape [n_step, n_class]
    auto output = model.transpose(0, 1);
    output = torch::log_softmax(output, 2);
    return output;
    // , trained_attn

  }

  // get attention weight one 'dec_output' with 'enc_outputs'
  torch::Tensor get_att_weight(torch::Tensor dec_output, torch::Tensor enc_outputs) {
    auto n_step = enc_outputs.sizes().vec().at(0);
    auto batch_size = enc_outputs.sizes().vec().at(1);
    auto attn_scores = torch::zeros({n_step, batch_size}).to(device);  // attn_scores : [n_step]

    for( int i = 0 ; i < n_step; i++ ) {
      attn_scores[i] = get_att_score(dec_output, enc_outputs[i]);
    }

    // Normalize scores to weights in range 0 to 1
    return at::softmax(attn_scores, 0).transpose(0,1).unsqueeze(1);
  }

  torch::Tensor get_att_score(torch::Tensor dec_output, torch::Tensor enc_output) {
    // enc_outputs [batch_size, num_directions(=1) * n_hidden]
    auto batch_size = enc_output.sizes().vec().at(0);
    auto score = attn->forward(enc_output); // score : [batch_size, n_hidden]
    // return torch::bdot(dec_output.view({batch_size, -1}), score.view({batch_size, -1}));  // inner product make scalar value
    auto a = dec_output.view({batch_size, -1}).unsqueeze(1);
    auto b = score.view({batch_size, -1}).unsqueeze(2);
    return a.bmm(b).squeeze();  // inner product make scalar value
  }

  int64_t n_class;
  torch::Device device;
  std::shared_ptr<torch::nn::RNNImpl> enc_cell;
  std::shared_ptr<torch::nn::RNNImpl> dec_cell;
  std::shared_ptr<torch::nn::LinearImpl> attn;
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
  device_type = torch::kCPU;
  torch::Device device(device_type);

  std::vector<std::pair<std::string, std::string>>
    sentences {{"ich mochte ein bier", "i want a beer"},
               {"du liebst ein bier", "you love a beer"}
   };

  const int64_t batch_size = static_cast<int64_t>(sentences.size());

  auto dataset = torch::data::datasets::NLP(sentences);
  auto train_dataset = dataset.map(torch::data::transforms::Stack<torch::data::RNNExample>());
  auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(train_dataset), batch_size);

  auto hidden = torch::Tensor(torch::zeros({1, batch_size, n_hidden})).to(device);
  int64_t nClass = dataset.getClassNumber();
  Net model(nClass, device);

  model.to(device);
  torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(0.001));
  model.train();


  for (size_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {

    float loss_value;

    for (auto& batch : *data_loader) {
      auto input = batch.data.first.to(device);
      auto output = batch.data.second.to(device);
      auto targets = batch.target.to(device);

      optimizer.zero_grad();
      auto predict = model.forward(input, hidden, output);

      torch::Tensor loss;
      for( int i = 0 ; i < batch_size; i++ ) {
        if( loss.defined() )
          loss += torch::nll_loss(predict[i], targets[i]);
        else
          loss = torch::nll_loss(predict[i], targets[i]);
      }

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

    auto input = batch.data.first.to(device);
    auto output = batch.data.second.to(device);
    auto batch_size = output.sizes().vec().at(0);
    auto n_step = output.sizes().vec().at(1);
    std::vector<std::string> output_sentences;
    for( int b = 0 ; b < batch_size ; b++ ) {

      std::string sentence = "S";
      for( int s = 1 ; s < n_step; s++ ) {
        sentence += " P";
      }
      output_sentences.push_back(sentence);
    }
    output = dataset.strings_to_tensor(output_sentences).to(device);;
    auto targets = batch.target.to(device);
    auto hidden = torch::Tensor(torch::zeros({1, batch_size, n_hidden})).to(device);
    auto predict = model.forward(input, hidden, output);

    input = input.argmax(2).cpu();
    targets = targets.cpu();
    predict = predict.argmax(2).cpu();

    auto input_accessor = input.accessor<int64_t,2>();
    auto targets_accessor = targets.accessor<int64_t,2>();
    auto predict_accessor = predict.accessor<int64_t,2>();

    std::cout << std::endl;
    for(int i = 0; i < input_accessor.size(0); i++) {
      for(int j = 0; j < input_accessor.size(1); j++) {
        std::cout << " ";
        std::cout << dataset.index_to_string(input_accessor[i][j]);
      }
      std::cout << ".";
      for(int j = 0; j < predict_accessor.size(1); j++) {
        std::cout << " ";
        std::cout << dataset.index_to_string(predict_accessor[i][j]);
      }

      std::cout << ". [";
      for(int j = 0; j < targets_accessor.size(1); j++) {
        std::cout << " ";
        std::cout << dataset.index_to_string(targets_accessor[i][j]);
      }
      std::cout << ". ]";

      std::cout << std::endl;

    }
  }
}


