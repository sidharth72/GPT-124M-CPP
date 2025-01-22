#include "scaled_dot_attention.hpp"
#include "activations.hpp"
#include <cmath>

ScaledDotAttention::ScaledDotAttention(float dropout_prob) 
    : dropout_probability(dropout_prob) {
    if (dropout_prob < 0.0f || dropout_prob >= 1.0f) {
        throw std::invalid_argument("Dropout probability must be in range [0, 1)");
    }
}

std::pair<xt::xarray<float>, xt::xarray<float>> ScaledDotAttention::forward(
    const xt::xarray<float>& query,
    const xt::xarray<float>& key,
    const xt::xarray<float>& value,
    const xt::xarray<float>* mask) {
    
    // Get dimensions
    auto batch_size = query.shape()[0];
    auto seq_len = query.shape()[1];
    auto d_k = query.shape()[2];

    // Transpose key for attention calculation
    auto key_transposed = xt::transpose(key, {0, 2, 1});
    
    // Initialize attention scores
    // The attention scores will be of shape [batch_size, seq_len, seq_len]
    // This is because we are computing pairwise attention scores for each token in the sequence
    // This forms a square matrix of size seq_len x seq_len where each score represents the attention
    // that the query token should pay to the key token.
    xt::xarray<float> attention_scores = xt::zeros<float>({batch_size, seq_len, seq_len});
    
    // Compute attention scores batch-wise
    for (size_t b = 0; b < batch_size; ++b) {
        auto q_slice = xt::view(query, b, xt::all(), xt::all());
        auto k_slice = xt::view(key_transposed, b, xt::all(), xt::all());
        xt::xarray<float> result_slice = xt::linalg::dot(xt::eval(q_slice), xt::eval(k_slice));
        xt::view(attention_scores, b, xt::all(), xt::all()) = result_slice;
    }

    // Scale attention scores
    float scale = std::sqrt(static_cast<float>(d_k));
    xt::xarray<float> scaled_attention_scores = xt::eval(attention_scores / scale);

    // Apply mask if provided
    if (mask != nullptr) {
        xt::xarray<float> mask_value = xt::ones_like(scaled_attention_scores) * 
            (-std::numeric_limits<float>::infinity());
        scaled_attention_scores = xt::eval(xt::where(
            xt::equal(*mask, 1.0f),
            mask_value,
            scaled_attention_scores
        ));
    }

    // Apply softmax to convert the scores to probabilities sum up to 1
    xt::xarray<float> attention_weights = activation::Softmax::forward(scaled_attention_scores, 2);

    // Apply dropout if needed
    if (dropout_probability > 0.0f) {
        xt::xarray<float> dropout_mask = 
            xt::cast<float>(xt::random::rand<float>(attention_weights.shape()) > dropout_probability);
        attention_weights = xt::eval(attention_weights * dropout_mask / (1.0f - dropout_probability));
    }
    
    // Compute output through batch matrix multiplication
    xt::xarray<float> output = xt::zeros<float>({batch_size, seq_len, value.shape()[2]});
    
    for (size_t b = 0; b < batch_size; ++b) {
        auto weights_slice = xt::view(attention_weights, b, xt::all(), xt::all());
        auto value_slice = xt::view(value, b, xt::all(), xt::all());
        xt::xarray<float> result_slice = xt::linalg::dot(weights_slice, value_slice);
        xt::view(output, b, xt::all(), xt::all()) = result_slice;
    }

    return std::make_pair(output, attention_weights);
}