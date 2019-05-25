
/********************************************
 *
 * $ midair build
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
const int64_t kNumberOfEpochs = 20;

// After how many batches to log a new update with the loss value.
const int64_t kLogInterval = 1;

// Bi-LSTM(Attention) Parameters
const int64_t d_model  = 512;  // Embedding Size
const int64_t d_ff     = 2048; // FeedForward dimension
const int64_t d_k      = 64;   // dimension of K(=Q)
const int64_t d_v      = 64;   // dimension of V(=Q)
const int64_t n_layers = 6;    // number of Encoder of Decoder Layer
const int64_t n_heads  = 8;    // number of heads in Multi-Head Attention

std::shared_ptr<torch::nn::LSTMImpl> make_lstm(int64_t input_size, int64_t hidden_size) {
  auto options = torch::nn::LSTMOptions(input_size, hidden_size);
  options.bidirectional(true);
  return std::make_shared<torch::nn::LSTMImpl>(options);
}

std::shared_ptr<torch::nn::LinearImpl> make_linear(int64_t input_size, int64_t hidden_size, bool with_bias=true) {
  auto options = torch::nn::LinearOptions(input_size, hidden_size);
  options.with_bias(with_bias);
  return std::make_shared<torch::nn::LinearImpl>(options);
}

torch::Tensor get_sinusoid_encoding_table(int n_position, int d_model) {
  auto cal_angle = [&](float position, float hid_idx) { return position / std::pow(10000.0, 2.0 * std::floor(hid_idx / 2.0) / (float)d_model); };
  auto get_posi_angle_vec = [&](int position)
                            {
                              std::vector<float> vec;
                              for(int hid_j = 0 ; hid_j < d_model ; hid_j++ ) {
                                auto angle = cal_angle(position, hid_j);
                                vec.push_back( hid_j % 2 == 0 ? sin(angle) : cos(angle) );
                              }
                              const int64_t count = vec.size();
                              torch::Tensor output = torch::empty({count}, torch::kFloat);
                              std::memcpy(output.data_ptr(), vec.data(), output.numel() * sizeof(float));
                              return output.unsqueeze(0);
                            };

  std::vector<torch::Tensor> vec;
  for(int pos_i = 0 ; pos_i < n_position ; pos_i++ ) {
    vec.push_back(get_posi_angle_vec(pos_i));
  }

  return torch::cat(vec);
}

torch::Tensor get_attn_pad_mask(torch::Tensor seq_q, torch::Tensor seq_k) {
  auto q_sizes = seq_q.sizes();
  auto k_sizes = seq_k.sizes();
  auto batch_size = q_sizes[0];
  auto len_q = q_sizes[1];
  auto len_k = k_sizes[1];
  // eq(zero) is PAD token
  auto pad_attn_mask = seq_k.eq(0).unsqueeze(1);  // batch_size x 1 x len_k(=len_q), one is masking
  return pad_attn_mask.expand({batch_size, len_q, len_k});  // batch_size x len_q x len_k
}


torch::Tensor get_attn_subsequent_mask(torch::Tensor seq) {
  auto attn_shape = {seq.size(0), seq.size(1), seq.size(1)};
  auto subsequent_mask = torch::ones(attn_shape).to(torch::kFloat).triu(1);
  return subsequent_mask;
}

std::shared_ptr<torch::nn::EmbeddingImpl> make_embedding(torch::Tensor weight) {
  auto sizes = weight.sizes().vec();
  auto emb = std::make_shared<torch::nn::EmbeddingImpl>(sizes.at(0), sizes.at(1));
  emb->weight = weight;
  return emb;
}

std::shared_ptr<torch::nn::Conv1dImpl> conv1d(int64_t input_channels, int64_t output_channels, torch::ExpandingArray<1> kernel_size) {
  auto options = torch::nn::Conv1dOptions(input_channels, output_channels, kernel_size).stride(1).with_bias(true);
  return std::make_shared<torch::nn::Conv1dImpl>(options);
}

std::shared_ptr<torch::nn::Conv2dImpl> conv2d(int64_t input_channels, int64_t output_channels, torch::ExpandingArray<2> kernel_size) {
  auto options = torch::nn::Conv2dOptions(input_channels, output_channels, kernel_size).stride(1).with_bias(true);
  return std::make_shared<torch::nn::Conv2dImpl>(options);
}


struct ScaledDotProductAttention : torch::nn::Module {
  ScaledDotProductAttention(torch::Device device)
    : device(device)
  {
  }

  std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor Q,
                                                  torch::Tensor K,
                                                  torch::Tensor V,
                                                  torch::Tensor attn_mask) {
    auto scores = torch::matmul(Q, K.transpose(-1, -2)) / std::sqrt(d_k); // scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
    scores.masked_fill_(attn_mask, -1e9); // Fills elements of self tensor with value where mask is one.
    auto attn = at::softmax(scores, -1);
    auto context = torch::matmul(attn, V);
    return std::make_pair(context, attn);
  }

  torch::Device device;
};


struct MultiHeadAttention : torch::nn::Module { //
  MultiHeadAttention(torch::Device device)
    : device(device)
    , W_Q(make_linear(d_model, d_k * n_heads))
    , W_K(make_linear(d_model, d_k * n_heads))
    , W_V(make_linear(d_model, d_k * n_heads))
  {
    register_module("W_Q", W_Q);
    register_module("W_K", W_K);
    register_module("W_V", W_V);
  }

  std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor Q,
                                                  torch::Tensor K,
                                                  torch::Tensor V,
                                                  torch::Tensor attn_mask) {
    // q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
    auto residual = Q;
    auto batch_size = Q.size(0);

    // (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
    auto q_s = W_Q->forward(Q).view({batch_size, -1, n_heads, d_k}).transpose(1,2); // q_s: [batch_size x n_heads x len_q x d_k]
    auto k_s = W_K->forward(K).view({batch_size, -1, n_heads, d_k}).transpose(1,2); // k_s: [batch_size x n_heads x len_k x d_k]
    auto v_s = W_V->forward(V).view({batch_size, -1, n_heads, d_v}).transpose(1,2); // v_s: [batch_size x n_heads x len_k x d_v]

    attn_mask = attn_mask.unsqueeze(1).repeat({1, n_heads, 1, 1}); // attn_mask : [batch_size x n_heads x len_q x len_k]

    // context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
    torch::Tensor context, attn;
    std::tie(context, attn) = ScaledDotProductAttention(device).forward(q_s, k_s, v_s, attn_mask);
    context = context.transpose(1, 2).contiguous().view({batch_size, -1, n_heads * d_v}); // context: [batch_size x len_q x n_heads * d_v]
    auto linear = make_linear(n_heads * d_v, d_model);
    linear->to(device);
    auto output = linear->forward(context);
    return std::make_pair(at::layer_norm(output + residual, {d_model}), attn); // output: [batch_size x len_q x d_model]
  }

  torch::Device device;
  std::shared_ptr<torch::nn::LinearImpl> W_Q;
  std::shared_ptr<torch::nn::LinearImpl> W_K;
  std::shared_ptr<torch::nn::LinearImpl> W_V;
};


struct PoswiseFeedForwardNet : torch::nn::Module {
  PoswiseFeedForwardNet(torch::Device device)
    : device(device)
    , conv1(conv1d(d_model, d_ff, {1}))
    , conv2(conv1d(d_ff, d_model, {1}))
  {
    register_module("conv1", conv1);
    register_module("conv2", conv2);
  }

  torch::Tensor forward(torch::Tensor inputs) {
    auto residual = inputs; // inputs : [batch_size, len_q, d_model]
    auto output = at::relu(conv1->forward(inputs.transpose(1, 2)));
    output = conv2->forward(output).transpose(1, 2);
    return at::layer_norm(output + residual, {d_model});
  }

  torch::Device device;
  std::shared_ptr<torch::nn::Conv1dImpl> conv1;
  std::shared_ptr<torch::nn::Conv1dImpl> conv2;
};


struct EncoderLayer : torch::nn::Module { //
  EncoderLayer(torch::Device device)
    : device(device)
    , enc_self_attn(std::make_shared<MultiHeadAttention>(device))
    , pos_ffn(std::make_shared<PoswiseFeedForwardNet>(device))
  {
    register_module("enc_self_attn", enc_self_attn);
    register_module("pos_ffn", pos_ffn);
  }

  std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor enc_inputs, torch::Tensor enc_self_attn_mask) {
    torch::Tensor enc_outputs, attn;
    std::tie(enc_outputs, attn) = enc_self_attn->forward(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask); // enc_inputs to same Q,K,V
    enc_outputs = pos_ffn->forward(enc_outputs); // enc_outputs: [batch_size x len_q x d_model]
    return std::make_pair(enc_outputs, attn);
  }

  torch::Device device;
  std::shared_ptr<MultiHeadAttention> enc_self_attn;
  std::shared_ptr<PoswiseFeedForwardNet> pos_ffn;
};

struct DecoderLayer : torch::nn::Module { //
  DecoderLayer(torch::Device device)
    : device(device)
    , dec_self_attn(std::make_shared<MultiHeadAttention>(device))
    , dec_enc_attn(std::make_shared<MultiHeadAttention>(device))
    , pos_ffn(std::make_shared<PoswiseFeedForwardNet>(device))
  {
    register_module("dec_self_attn", dec_self_attn);
    register_module("dec_enc_attn", dec_enc_attn);
    register_module("pos_ffn", pos_ffn);
  }

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor dec_inputs, torch::Tensor enc_outputs, torch::Tensor dec_self_attn_mask, torch::Tensor dec_enc_attn_mask) {

    torch::Tensor dec_outputs, dec_self_attn_, dec_enc_attn_;
    std::tie(dec_outputs, dec_self_attn_) = dec_self_attn->forward(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask);
    std::tie(dec_outputs, dec_enc_attn_) = dec_enc_attn->forward(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask);
    dec_outputs = pos_ffn->forward(dec_outputs);
    return std::make_tuple(dec_outputs, dec_self_attn_, dec_enc_attn_);
  }

  torch::Device device;
  std::shared_ptr<MultiHeadAttention> dec_self_attn;
  std::shared_ptr<MultiHeadAttention> dec_enc_attn;
  std::shared_ptr<PoswiseFeedForwardNet> pos_ffn;
};


struct Encoder : torch::nn::Module { //
  Encoder(int64_t src_vocab_size, torch::Device device)
    : device(device)
    , src_emb(std::make_shared<torch::nn::EmbeddingImpl>(src_vocab_size, d_model))
    , pos_emb(make_embedding(get_sinusoid_encoding_table(src_vocab_size, d_model).to(device)))
    , layers(std::make_shared<torch::nn::SequentialImpl>())
  {
    register_module("src_emb", src_emb);
    register_module("pos_emb", pos_emb);
    for( int i = 0 ; i < n_layers ; i++ )
      layers->push_back(std::make_shared<EncoderLayer>(device));
    register_module("layers", layers);
  }

  std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor enc_inputs) {

    torch::Tensor long_tensor = torch::empty({1, 5}, torch::kInt64);
    int64_t values[] = {1,2,3,4,0};
    std::memcpy(long_tensor.data_ptr(), values, long_tensor.numel() * sizeof(int64_t));
    long_tensor = long_tensor.to(at::kLong).to(device);

    auto enc_outputs = src_emb->forward(enc_inputs) + pos_emb->forward(long_tensor);
    auto enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs);
    std::vector<torch::Tensor> enc_self_attns;

    for( auto layer : *layers ) {
      torch::Tensor enc_self_attn;
      std::tie(enc_outputs, enc_self_attn) = layer.forward<std::pair<torch::Tensor, torch::Tensor>>(enc_outputs, enc_self_attn_mask);
      enc_self_attns.push_back(enc_self_attn);
    }

    return std::make_pair(enc_outputs, at::cat(enc_self_attns));
  }

  torch::Device device;
  std::shared_ptr<torch::nn::EmbeddingImpl> src_emb;
  std::shared_ptr<torch::nn::EmbeddingImpl> pos_emb;
  std::shared_ptr<torch::nn::SequentialImpl> layers;
};

struct Decoder : torch::nn::Module {
  Decoder(int64_t tgt_vocab_size, torch::Device device)
    : device(device)
    , tgt_emb(std::make_shared<torch::nn::EmbeddingImpl>(tgt_vocab_size, d_model))
    , pos_emb(make_embedding(get_sinusoid_encoding_table(tgt_vocab_size, d_model).to(device)))
    , layers(std::make_shared<torch::nn::SequentialImpl>())
  {
    register_module("tgt_emb", tgt_emb);
    register_module("pos_emb", pos_emb);
    for( int i = 0 ; i < n_layers ; i++ )
      layers->push_back(std::make_shared<DecoderLayer>(device));
    register_module("layers", layers);
  }

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor dec_inputs, torch::Tensor enc_inputs, torch::Tensor enc_outputs) {

    torch::Tensor long_tensor = torch::empty({1, 5}, torch::kInt64);
    int64_t values[] = {5,1,2,3,4};
    std::memcpy(long_tensor.data_ptr(), values, long_tensor.numel() * sizeof(int64_t));
    long_tensor = long_tensor.to(at::kLong).to(device);

    auto dec_outputs = tgt_emb->forward(dec_inputs) + pos_emb->forward(long_tensor);
    auto dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs).to(at::kFloat);
    auto dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs).to(device);
    // std::cout << dec_self_attn_pad_mask.to(at::kFloat) << std::endl;
    // std::cout << dec_self_attn_subsequent_mask.to(at::kFloat) << std::endl;
    // std::cout << dec_self_attn_pad_mask.dtype() << std::endl;
    // std::cout << dec_self_attn_subsequent_mask.dtype() << std::endl;
    // std::cout << dec_self_attn_subsequent_mask.to(at::kFloat).dtype() << std::endl;
    auto dec_self_attn_mask = at::gt(dec_self_attn_pad_mask + dec_self_attn_subsequent_mask, 0);

    auto dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs);

    std::vector<torch::Tensor> dec_self_attns, dec_enc_attns;
    for( auto layer : *layers ) {
      torch::Tensor dec_self_attn, dec_enc_attn;
      std::tie(dec_outputs, dec_self_attn, dec_enc_attn) =
        layer.forward
        <std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>>
        (dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask);

      dec_self_attns.push_back(dec_self_attn);
      dec_enc_attns.push_back(dec_enc_attn);
    }

    return std::make_tuple(dec_outputs, at::cat(dec_self_attns), at::cat(dec_enc_attns));
  }

  torch::Device device;
  std::shared_ptr<torch::nn::EmbeddingImpl> tgt_emb;
  std::shared_ptr<torch::nn::EmbeddingImpl> pos_emb;
  std::shared_ptr<torch::nn::SequentialImpl> layers;
};

struct Transformer : torch::nn::Module {

  Transformer(int64_t src_vocab_size, int64_t tgt_vocab_size, torch::Device device)
    : device(device),
      encoder(std::make_shared<Encoder>(src_vocab_size, device)),
      decoder(std::make_shared<Decoder>(tgt_vocab_size, device)),
      projection(make_linear(d_model, tgt_vocab_size, false))
  {
    register_module("encoder", encoder);
    register_module("decoder", decoder);
    register_module("projection", projection);
   }

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor enc_inputs, torch::Tensor dec_inputs) {
    torch::Tensor enc_outputs, enc_self_attns;
    std::tie(enc_outputs, enc_self_attns) = encoder->forward(enc_inputs);

    torch::Tensor dec_outputs, dec_self_attns, dec_enc_attns;
    std::tie(dec_outputs, dec_self_attns, dec_enc_attns) = decoder->forward(dec_inputs, enc_inputs, enc_outputs);
    auto dec_logits = projection->forward(dec_outputs); // dec_logits : [batch_size x src_vocab_size x tgt_vocab_size]
    dec_logits = dec_logits.view({-1, dec_logits.size(-1)});
    dec_logits = torch::log_softmax(dec_logits, /*dim=*/1);
    return std::make_tuple(dec_logits, enc_self_attns, dec_self_attns, dec_enc_attns);
  }

  torch::Device device;
  std::shared_ptr<Encoder> encoder;
  std::shared_ptr<Decoder> decoder;
  std::shared_ptr<torch::nn::LinearImpl> projection;
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
    sentences {{"ich mochte ein bier", "i want a beer"}
  };

  const int64_t batch_size = static_cast<int64_t>(sentences.size());
  auto dataset = torch::data::datasets::NLP(sentences);
  auto train_dataset = dataset.map(torch::data::transforms::Stack<torch::data::RNNExample>());
  auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(train_dataset), kTrainBatchSize);

  int64_t nClass = dataset.getClassNumber();
  Transformer model(nClass, nClass, device);
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
    torch::Tensor predict;
    std::tie(predict, std::ignore, std::ignore, std::ignore) = model.forward(input, output);

    torch::Tensor loss = torch::nll_loss(predict, targets.view({-1}));
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
    auto targets = batch.target.to(device);
    torch::Tensor predict;
    std::tie(predict, std::ignore, std::ignore, std::ignore) = model.forward(input, output);

    input = input.cpu();
    targets = targets.cpu();
    predict = predict.argmax(1).cpu().unsqueeze(0);

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


