// activation.hpp
#pragma once
#include <xtensor/xarray.hpp>
#include <xtensor/xmath.hpp>

namespace activation {

class ReLU {
public:
    static xt::xarray<float> forward(const xt::xarray<float>& input);
};

class GELU {
public:
    static xt::xarray<float> forward(const xt::xarray<float>& input);
private:
    static constexpr float sqrt_2_pi = 2.506628275f;  // √(2π)
};

class Softmax {
public:
    static xt::xarray<float> forward(const xt::xarray<float>& input, size_t axis);

    // static xt::xarray<float> forward_masked(
    //     const xt::xarray<float>& input,
    //     const xt::xarray<float>& mask,
    //     size_t axis
    // );

    static xt::xarray<float> apply_softmax(
        const xt::xarray<float>& input,
        const xt::xarray<float>& max_vals,
        size_t axis
    );
};

} // namespace nn
