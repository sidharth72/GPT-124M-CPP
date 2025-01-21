#include "multihead_self_attention.hpp"
#include <xtensor/xview.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <stdexcept>


MultiHeadAttention::MultiHeadAttention (
    size_t num_heads,
    size_t d_model,
    size_t d_k,
    size_t d_v,
    float dropout_prob
) : num_heads(num_heads), d_model(d_model), d_k(d_k), d_v(d_v), attention(dropout_prob) {
    if (d_model % num_heads != 0) {
        throw std::invalid_argument("d_model must be divisible by num_heads");
    }
}

std::tuple<xt::xarray<float>, xt::xarray<float>, xt::xarray<float>>
MultiHeadAttention::split_qkv(const xt::xarray<float>& projected) {
    // projected shape: [batch_size, seq_len, 3 * d_model]
    auto batch_size = projected.shape()[0];
    auto seq_len = projected.shape()[1];
    
    // Split into Q, K, V
    xt::xarray<float> q = xt::view(projected, xt::all(), xt::all(), xt::range(0, d_model));
    xt::xarray<float> k = xt::view(projected, xt::all(), xt::all(), xt::range(d_model, 2 * d_model));
    xt::xarray<float> v = xt::view(projected, xt::all(), xt::all(), xt::range(2 * d_model, 3 * d_model));
    
    return std::make_tuple(q, k, v);
}

std::vector<xt::xarray<float>> MultiHeadAttention::split_heads(const xt::xarray<float>& x) {
    // x shape: [batch_size, seq_len, d_model]
    auto batch_size = x.shape()[0];
    auto seq_len = x.shape()[1];
    size_t depth = d_model / num_heads;
    
    std::vector<xt::xarray<float>> heads;
    heads.reserve(num_heads); // Reserve space for the heads, this can be used for optimizing the dynamic memory allocation which will slow down the program
    
    // split into multiple heads, 12 in case of GPT
    for (size_t i = 0; i < num_heads; ++i) {
        xt::xarray<float> head = xt::view(x, xt::all(), xt::all(), xt::range(i * depth, (i + 1) * depth));
        heads.push_back(head);
    }
    
    return heads;
}


xt::xarray<float> MultiHeadAttention::combine_heads(const std::vector<xt::xarray<float>>& heads) {
    auto batch_size = heads[0].shape()[0];
    auto seq_len = heads[0].shape()[1];
    
    std::vector<size_t> new_shape = {batch_size, seq_len, d_model};
    xt::xarray<float> combined = xt::zeros<float>(new_shape);
    
    size_t depth = d_model / num_heads;
    for (size_t i = 0; i < num_heads; ++i) {
        // Directly assign to the view using xt::strided_view
        xt::strided_view(combined, {xt::all(), xt::all(), xt::range(i * depth, (i + 1) * depth)}) = heads[i];
    }
    
    return combined;
}


xt::xarray<float> MultiHeadAttention::forward(
    const xt::xarray<float>& input,
    const xt::xarray<float>& weights,
    const xt::xarray<float>& projection_weights,
    const xt::xarray<float>& biases,
    const xt::xarray<float>& projection_biases,
    const xt::xarray<float>* mask // Optional parameter
) {
    auto batch_size = input.shape()[0];
    auto seq_len = input.shape()[1];
    
    // Project input to Q, K, V space
    xt::xarray<float> projected = xt::linalg::dot(input, weights) + biases;
    
    // Split the projected matrix into Q, K, V
    xt::xarray<float> q, k, v;
    std::tie(q, k, v) = this->split_qkv(projected);
    
    // Split each of Q, K, V into heads
    auto q_heads = this->split_heads(q);
    auto k_heads = this->split_heads(k);
    auto v_heads = this->split_heads(v);
    
    std::vector<xt::xarray<float>> attention_outputs;
    attention_outputs.reserve(this->num_heads);
    
    size_t head_dim = d_model / num_heads;
    
    // Apply attention for each head
    for (size_t i = 0; i < this->num_heads; ++i) {
        // Ensure proper shapes for attention calculation
        auto q_head = xt::eval(xt::reshape_view(q_heads[i], {batch_size, seq_len, head_dim}));
        auto k_head = xt::eval(xt::reshape_view(k_heads[i], {batch_size, seq_len, head_dim}));
        auto v_head = xt::eval(xt::reshape_view(v_heads[i], {batch_size, seq_len, head_dim}));
        
        auto attention_result = this->attention.forward(q_head, k_head, v_head, mask);
        attention_outputs.push_back(attention_result.first);
    }
    
    // Combine the attention outputs
    auto combined_attention = this->combine_heads(attention_outputs);
    
    // Final projection
    xt::xarray<float> output = xt::linalg::dot(combined_attention, projection_weights) + projection_biases;
    
    return output;
}