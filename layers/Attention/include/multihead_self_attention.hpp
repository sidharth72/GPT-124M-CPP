#pragma once
#include "scaled_dot_attention.hpp"
#include <xtensor/xarray.hpp>
#include <vector>

class MultiHeadAttention {
public:

    // MultiHeadAttention constructor which receives
    // the number of heads, the model dimension, the key dimension, value dimension, etc
    MultiHeadAttention(
        size_t num_heads,
        size_t d_model,
        size_t d_k,
        size_t d_v,
        float dropout_prob = 0.0f
    );

    xt::xarray<float> forward(
        const xt::xarray<float>& input,
        const xt::xarray<float>& weights,
        const xt::xarray<float>& projection_weights,
        const xt::xarray<float>& biases,
        const xt::xarray<float>& projection_biases,
        const xt::xarray<float>* mask = nullptr
    );

private:
    size_t num_heads;
    size_t d_model;
    size_t d_k;
    size_t d_v;
    ScaledDotAttention attention; // ScaledDotAttention object which we will use in the later implementation
    
    // Helper functions for splitting and combining heads

    // The split heads function will return a tuple of xarrays that contains
    // the split heads of the input xarray
    std::vector<xt::xarray<float>> split_heads(const xt::xarray<float>& x);

    // The combine heads function will return a xarray that contains
    // the combined heads of the input xarray
    xt::xarray<float> combine_heads(const std::vector<xt::xarray<float>>& x);
    
    // Helper function to split QKV
    std::tuple<xt::xarray<float>, xt::xarray<float>, xt::xarray<float>> 
    split_qkv(const xt::xarray<float>& projected);
};