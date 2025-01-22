#ifndef MLP_HPP
#define MLP_HPP

#include <xtensor/xarray.hpp>

class MLP {
public:
    // Constructor only needs dropout probability
    explicit MLP(float dropout_prob = 0.0f);
    
    // Forward pass method - dimensions determined by input weights
    xt::xarray<float> forward(
        const xt::xarray<float>& input,
        const xt::xarray<float>& fc1_weights,
        const xt::xarray<float>& fc1_bias,
        const xt::xarray<float>& fc2_weights,
        const xt::xarray<float>& fc2_bias
    );

private:
    float dropout_prob;
    bool training{true}; // Training mode flag
    
    // Helper method for applying dropout
    xt::xarray<float> apply_dropout(const xt::xarray<float>& x);
};

#endif // MLP_HPP