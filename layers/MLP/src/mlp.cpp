#include "mlp.hpp"
#include "activations.hpp"
#include <xtensor-blas/xlinalg.hpp>
#include <stdexcept>
#include <xtensor/xrandom.hpp>

MLP::MLP(float dropout_prob) : dropout_prob(dropout_prob) {
    if (dropout_prob < 0.0f || dropout_prob >= 1.0f) {
        throw std::invalid_argument("Dropout probability must be in range [0, 1)");
    }
}

xt::xarray<float> MLP::apply_dropout(const xt::xarray<float>& x) {
    if (dropout_prob == 0.0f || !training) {
        return x;
    }
    
    auto dropout_mask = xt::cast<float>(
        xt::random::rand<float>(x.shape()) > dropout_prob
    );
    
    return (x * dropout_mask) / (1.0f - dropout_prob);
}

xt::xarray<float> MLP::forward(
    const xt::xarray<float>& input,
    const xt::xarray<float>& fc1_weights,
    const xt::xarray<float>& fc1_bias,
    const xt::xarray<float>& fc2_weights,
    const xt::xarray<float>& fc2_bias
) {
    // First linear layer with GELU activation
    auto h = xt::linalg::dot(input, fc1_weights);
    h = h + fc1_bias;
    h = activation::GELU::forward(h);

    // No need to apply droupout during inference
    //h = apply_dropout(h);
    
    // Second linear layer
    auto output = xt::linalg::dot(h, fc2_weights);
    output = output + fc2_bias;
    //output = apply_dropout(output);
    
    return output;
}