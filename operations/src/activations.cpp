// activations.cpp

#include "activations.hpp"
#include <cmath>

namespace activation {

    xt::xarray<float> ReLU::forward(const xt::xarray<float>& input) {
        return xt::maximum(input, 0.0f);
    }

    xt::xarray<float> GELU::forward(const xt::xarray<float>& input) {
        // GELU(x) = x * Φ(x)
        // where Φ(x) is the cumulative distribution function of the standard normal distribution
        // We use the approximation: GELU(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
        
        auto x3 = xt::pow(input, 3);
        auto inner = 0.797884f * (input + 0.044715f * x3);  // √(2/π) ≈ 0.797884
        auto tanh_inner = xt::tanh(inner);
        
        return 0.5f * input * (1.0f + tanh_inner);
    }

    xt::xarray<float> Softmax::forward(const xt::xarray<float>& input, size_t axis) {
        // Find max values along specified axis for numerical stability
        auto max_vals = xt::amax(input, {axis}, xt::keep_dims);
        return apply_softmax(input, max_vals, axis);
    }

    // xt::xarray<float> Softmax::forward_masked(
    //     const xt::xarray<float>& input,
    //     const xt::xarray<float>& mask,
    //     size_t axis) {
        
    //     // Apply mask by setting masked positions to large negative values
    //     auto masked_input = xt::where(
    //         xt::equal(mask, 0.0f),
    //         xt::ones_like(input) * -std::numeric_limits<float>::infinity(),
    //         input
    //     );
        
    //     // Find max values along specified axis for numerical stability
    //     auto max_vals = xt::amax(masked_input, {axis}, xt::keep_dims);
    //     return apply_softmax(masked_input, max_vals, axis);
    // }

    xt::xarray<float> Softmax::apply_softmax(
        const xt::xarray<float>& input,
        const xt::xarray<float>& max_vals,
        size_t axis) {
        
        // Subtract max values for numerical stability
        auto exp_values = xt::exp(input - max_vals);
        
        // Sum along specified axis
        auto sum_exp = xt::sum(exp_values, {axis}, xt::keep_dims);
        
        // Compute softmax
        return exp_values / sum_exp;
    }
}