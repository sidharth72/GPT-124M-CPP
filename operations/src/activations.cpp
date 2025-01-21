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
        // Explicitly evaluate max values and keep in memory
        auto max_vals = xt::eval(xt::amax(input, {axis}, xt::keep_dims));
        return apply_softmax(input, max_vals, axis);
    }

    xt::xarray<float> Softmax::apply_softmax(
        const xt::xarray<float>& input,
        const xt::xarray<float>& max_vals,
        size_t axis) {
        
        // Create a temporary array for shifted values
        auto shifted = xt::eval(input - max_vals);
        
        // Compute exponentials with explicit evaluation
        auto exp_values = xt::eval(xt::exp(shifted));
        
        // Ensure sum is computed and stored
        auto sum_exp = xt::eval(xt::sum(exp_values, {axis}, xt::keep_dims));
        
        // Guard against division by zero
        auto safe_sum = xt::maximum(sum_exp, std::numeric_limits<float>::epsilon());
        
        // Compute final result with explicit evaluation
        return xt::eval(exp_values / safe_sum);
    }
}