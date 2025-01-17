#include "scaled_dot_attention.hpp"
#include "activations.hpp"
#include <cmath>

ScaledDotAttention::ScaledDotAttention(float dropout_prob) : dropout_probability(dropout_prob) {
    if (dropout_prob < 0.0f || dropout_prob >= 1.0f) {
        throw std::invalid_argument("Dropout probability must be in range [0, 1)");
    }
}

std::pair<xt::xarray<float>, xt::xarray<float>> ScaledDotAttention::forward(
    const xt::xarray<float>& query,
    const xt::xarray<float>& key,
    const xt::xarray<float>& value,
    const xt::xarray<float>* mask) {
    
    // Compute attention scores
    // Eguation: attention_scores = Q * K^T
    auto attention_scores = xt::linalg::dot(query, xt::transpose(key));
    
    // Scale the attention scores
    float scale = std::sqrt(static_cast<float>(key.shape()[1]));  // sqrt(d_k)
    auto scaled_attention_scores = xt::eval(attention_scores / scale);  // Added xt::eval()

    
    // Apply softmax (with or without mask)
    xt::xarray<float> attention_weights;

    if (mask != nullptr) {
        // Apply the mask
        scaled_attention_scores = xt::where(
            xt::equal(*mask, 0.0f),
            xt::ones_like(scaled_attention_scores) * -std::numeric_limits<float>::infinity(),
            scaled_attention_scores
        );
  

    }
    // Applying the softmax
    attention_weights = activation::Softmax::forward(scaled_attention_scores, 1);
    
    // Apply dropout
    if (dropout_probability > 0.0f) {
        auto dropout_mask = xt::random::rand<float>(attention_weights.shape()) > dropout_probability;
        attention_weights *= xt::cast<float>(dropout_mask) / (1.0f - dropout_probability);
    }
    
    // Compute output
    auto output = xt::linalg::dot(attention_weights, value);
    
    return std::make_pair(output, attention_weights);
}