

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
const int64_t kNumberOfEpochs = 100;

// After how many batches to log a new update with the loss value.
const int64_t kLogInterval = 10;


// BERT Parameters
const int64_t maxlen = 30;
const int64_t max_pred = 5; // max tokens of prediction
const int64_t n_layers = 6;
const int64_t n_heads = 12;
const int64_t d_model = 768;
const int64_t d_ff = 768*4; // 4*d_model, FeedForward dimension
const int64_t d_k = 64; // dimension of K(=Q)
const int64_t d_v = 64; // dimension of V(=Q)
const int64_t n_segments = 2;


torch::Tensor gelu(torch::Tensor x) {
  // Implementation of the gelu activation function by Hugging Face
  return x * 0.5 * (1.0 + torch::erf( x / std::sqrt(2.0)));
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

std::shared_ptr<torch::nn::LinearImpl> make_linear(int64_t input_size, int64_t hidden_size, bool with_bias=true) {
  auto options = torch::nn::LinearOptions(input_size, hidden_size);
  options.with_bias(with_bias);
  return std::make_shared<torch::nn::LinearImpl>(options);
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

struct Embedding : torch::nn::Module {
  Embedding(int64_t vocab_size, torch::Device device)
    : device(device),
      tok_embed(std::make_shared<torch::nn::EmbeddingImpl>(vocab_size, d_model)),  // token embedding
      pos_embed(std::make_shared<torch::nn::EmbeddingImpl>(maxlen, d_model)),  // position embedding
      seg_embed(std::make_shared<torch::nn::EmbeddingImpl>(n_segments, d_model))  // segment(token type) embedding
  {
    register_module("tok_embed", tok_embed);
    register_module("pos_embed", pos_embed);
    register_module("seg_embed", seg_embed);
  }

  torch::Tensor forward(torch::Tensor x, torch::Tensor seg) {
    auto seq_len = x.sizes().vec().at(1);
    auto pos = torch::arange(seq_len, torch::kLong).to(device);
    pos = pos.unsqueeze(0).expand_as(x); // (seq_len,) -> (batch_size, seq_len)
    auto embedding = tok_embed->forward(x) + pos_embed->forward(pos) + seg_embed->forward(seg);
    return at::layer_norm(embedding, {d_model});
  }

  torch::Device device;
  std::shared_ptr<torch::nn::EmbeddingImpl> tok_embed;
  std::shared_ptr<torch::nn::EmbeddingImpl> pos_embed;
  std::shared_ptr<torch::nn::EmbeddingImpl> seg_embed;

};


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
    , fc1(make_linear(d_model, d_ff))
    , fc2(make_linear(d_ff, d_model))
  {
    register_module("fc1", fc1);
    register_module("fc2", fc2);
  }

  torch::Tensor forward(torch::Tensor x) {
    // (batch_size, len_seq, d_model) -> (batch_size, len_seq, d_ff) -> (batch_size, len_seq, d_model)
    return fc2->forward(gelu(fc1->forward(x)));
  }

  torch::Device device;
  std::shared_ptr<torch::nn::LinearImpl> fc1;
  std::shared_ptr<torch::nn::LinearImpl> fc2;
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

struct BERT : torch::nn::Module {

  BERT(int64_t vocab_size, torch::Device device)
    : device(device),
      embedding(std::make_shared<Embedding>(vocab_size, device)),
      fc(make_linear(d_model, d_model)),
      layers(std::make_shared<torch::nn::SequentialImpl>()),
      linear(make_linear(d_model, d_model)),
      classifier(make_linear(d_model, 2))
      // decoder is shared with embedding layer
      // embed_weight(embedding->tok_embed->weight),
      // n_vocab(embed_weight.sizes().vec().at(0)),
      // n_dim(embed_weight.sizes().vec().at(1)),
      // decoder(make_linear(n_dim, n_vocab, false)),
      // decoder_bias(torch::zeros(n_vocab))
  {
    // decoder is shared with embedding layer
    embed_weight = embedding->tok_embed->weight;
    n_vocab = embedding->tok_embed->weight.sizes().vec().at(0);
    n_dim = embedding->tok_embed->weight.sizes().vec().at(1);
    decoder = make_linear(n_dim, n_vocab, false);
    decoder_bias = torch::zeros(n_vocab);
    for( int i = 0 ; i < n_layers ; i++ ) {
      layers->push_back(std::make_shared<EncoderLayer>(device));
    }
    decoder->weight = embedding->tok_embed->weight;
    register_module("embedding", embedding);
    register_module("fc", fc);
    register_module("linear", linear);
    register_module("classifier", classifier);
    register_module("decoder", decoder);
    register_module("layers", layers);
    register_parameter("embed_weight", embed_weight);
    register_parameter("decoder_bias", decoder_bias);
  }

  std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor input_ids, torch::Tensor segment_ids, torch::Tensor masked_pos) {

    torch::Tensor enc_self_attn_mask;
    auto output = embedding->forward(input_ids, segment_ids);
    enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids);
    for( auto layer : *layers ) {
      torch::Tensor enc_self_attn;
      std::tie(output, enc_self_attn) = layer.forward<std::pair<torch::Tensor, torch::Tensor>>(output, enc_self_attn_mask);
    }
    // output : [batch_size, len, d_model], attn : [batch_size, n_heads, d_mode, d_model]
    // it will be decided by first token(CLS)
    auto h_pooled = at::tanh(fc->forward(output.slice(1,0,1).squeeze())); // [batch_size, d_model]
    auto logits_clsf = classifier->forward(h_pooled); // [batch_size, 2]

    auto vec = masked_pos.sizes().vec();
    vec.push_back(1);
    auto sizes = torch::IntArrayRef(vec);
    masked_pos = masked_pos.view(sizes).expand({-1, -1, output.size(-1)}); // [batch_size, maxlen, d_model]
    auto h_masked = torch::gather(output, 1, masked_pos); // masking position [batch_size, len, d_model]
    h_masked = at::layer_norm(gelu(linear->forward(h_masked)), {d_model});
    auto logits_lm = decoder->forward(h_masked) + decoder_bias; //[batch_size, maxlen, n_vocab]

    logits_lm = torch::log_softmax(logits_lm, /*dim=*/1);
    logits_clsf = torch::log_softmax(logits_clsf, /*dim=*/1);
    return std::make_pair(logits_lm, logits_clsf);
  }

  torch::Device device;
  std::shared_ptr<Embedding> embedding;
  std::shared_ptr<torch::nn::LinearImpl> fc;
  std::shared_ptr<torch::nn::LinearImpl> linear;
  std::shared_ptr<torch::nn::LinearImpl> classifier;
  std::shared_ptr<torch::nn::LinearImpl> decoder;
  std::shared_ptr<torch::nn::SequentialImpl> layers;
  torch::Tensor embed_weight;
  torch::Tensor decoder_bias;
  int64_t n_vocab;
  int64_t n_dim;

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

  std::vector<std::string>
    sentences {"Hello, how are you? I am Romeo.\n"
               "Hello, Romeo My name is Juliet. Nice to meet you.\n"
               "Nice meet you too. How are you today?\n"
               "Great. My baseball team won the competition.\n"
               "Oh Congratulations, Juliet\n"
               "Thanks you Romeo"};

  auto dataset = torch::data::datasets::NLP(sentences);
  auto train_dataset = dataset.map(torch::data::transforms::Stack<torch::data::BERTExample>());
  auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(train_dataset), kTrainBatchSize);

  int64_t nClass = dataset.getClassNumber();
  BERT model(nClass, device);
  model.to(device);

  torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(0.001));
  model.train();

  for (size_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {

    float loss_value;

    for (auto& batch : *data_loader) {

      auto input_ids = std::get<0>(batch.data).to(device);
      auto segment_ids = std::get<1>(batch.data).to(device);
      auto masked_tokens = std::get<2>(batch.data).to(device);
      auto masked_pos = std::get<3>(batch.data).to(device);
      auto isNext = batch.target.to(device);
      int64_t batch_size = batch.target.sizes()[0];

      optimizer.zero_grad();
      torch::Tensor logits_lm, logits_clsf;
      std::tie(logits_lm, logits_clsf) = model.forward(input_ids, segment_ids, masked_pos);

      // std::cout << input_ids << std::endl;
      // std::cout << segment_ids << std::endl;
      // std::cout << masked_pos << std::endl;
      // std::cout << masked_tokens << std::endl;
      // std::cout << isNext << std::endl;

      torch::Tensor loss_lm; // for masked LM
      torch::Tensor loss_clsf; // for sentence classification

      for( int i = 0 ; i < batch_size; i++ ) {
        if( loss_lm.defined() )
          loss_lm += torch::nll_loss(logits_lm[i], masked_tokens[i]);
        else
          loss_lm = torch::nll_loss(logits_lm[i], masked_tokens[i]);

        if( loss_clsf.defined() )
          loss_clsf += torch::nll_loss(logits_clsf[i], isNext[i]);
        else
          loss_clsf = torch::nll_loss(logits_clsf[i], isNext[i]);
      }
      loss_lm = loss_lm.to(torch::kFloat).mean();

      torch::Tensor loss = loss_lm + loss_clsf;

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

    auto input_ids = std::get<0>(batch.data).to(device);
    auto segment_ids = std::get<1>(batch.data).to(device);
    auto masked_tokens = std::get<2>(batch.data).to(device);
    auto masked_pos = std::get<3>(batch.data).to(device);
    auto isNext = batch.target.to(device);
    int64_t batch_size = batch.target.sizes()[0];

    torch::Tensor predict;
    std::tie(predict, std::ignore) = model.forward(input_ids, segment_ids, masked_pos);

    auto input = input_ids.cpu();
    auto targets = segment_ids.cpu();
    predict = predict.argmax(1).cpu();

    auto input_accessor = input.accessor<int64_t,2>();
    auto targets_accessor = targets.accessor<int64_t,2>();
    auto predict_accessor = predict.accessor<int64_t,2>();

    std::cout << std::endl;
    for(int i = 0; i < input_accessor.size(0); i++) {
      for(int j = 0; j < input_accessor.size(1); j++) {
        std::cout << " ";
        std::cout << dataset.index_to_string(input_accessor[i][j]);
      }
      std::cout << "." << std::endl;
      for(int j = 0; j < predict_accessor.size(1); j++) {
        std::cout << " ";
        std::cout << dataset.index_to_string(predict_accessor[i][j]);
      }

      std::cout << "." << std::endl;
      std::cout << "[";
      for(int j = 0; j < targets_accessor.size(1); j++) {
        std::cout << " ";
        std::cout << dataset.index_to_string(targets_accessor[i][j]);
      }
      std::cout << ". ]" << std::endl;

      std::cout << std::endl;

    }
  }

}


