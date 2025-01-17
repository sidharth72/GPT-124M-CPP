/*

    Layer Normalization
    -------------------

    Layer normalization is a type of normalization technique that normalizes the activations of a layer for each individual sample. 
    It is similar to batch normalization, but instead of normalizing the activations across the entire batch, it normalizes the activations across the features.

    The formula for layer normalization is given by:

    y = gamma * (x - mean) / sqrt(variance + epsilon) + beta

    where:
    - x is the input tensor
    - gamma and beta are learnable parameters
    - mean and variance are the mean and variance of the input tensor
    - epsilon is a small value to prevent division by zero

    The mean and variance are computed along the last axis of the input tensor.

    The forward function of the layer normalization layer is implemented below.


*/




#include "layer_normalization.hpp"

LayerNormalization::LayerNormalization(const xt::xarray<float>& gamma, const xt::xarray<float>& beta, float eps) 
    : weight(gamma), bias(beta), epsilon(eps) {
    // Weights and biases are now passed from outside
}

xt::xarray<float> LayerNormalization::forward(const xt::xarray<float>& x) {
    // Get the last axis for reduction
    std::vector<std::size_t> axes = {x.dimension() - 1};
    
    // ===== The mean and variance tensor will have the same shape as the number of sequences =====

    // Compute mean along the last dimension
    xt::xarray<float> mean = xt::mean(x, axes, xt::keep_dims);
    std::cout << "Mean: " << xt::adapt(mean.shape()) << std::endl;
    
    // Compute standard deviation along the last dimension
    xt::xarray<float> variance = xt::mean(xt::square(x - mean), axes, xt::keep_dims);
    xt::xarray<float> std_dev = xt::sqrt(variance + epsilon);
    
    // Normalize
    xt::xarray<float> normalized = (x - mean) / std_dev;

    // Reshape weight and bias for broadcasting for supporting input dimensions
    std::vector<std::size_t> new_shape(x.dimension(), 1);
    new_shape.back() = weight.shape(0);  // Set last dimension to match weight/bias size
    
    xt::xarray<float> weight_broadcasted = xt::reshape_view(weight, new_shape);
    xt::xarray<float>  bias_broadcasted = xt::reshape_view(bias, new_shape);
    
    // Scale and shift
    return normalized * weight_broadcasted + bias_broadcasted;
}