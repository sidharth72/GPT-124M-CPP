// scaled_dot_attention.hpp
#pragma once
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xeval.hpp>  // Add this for xt::eval
#include <xtensor/xadapt.hpp>

class ScaledDotAttention {
public:
    explicit ScaledDotAttention(float dropout_prob = 0.0f);
    
    std::pair<xt::xarray<float>, xt::xarray<float>> forward(
        const xt::xarray<float>& query,
        const xt::xarray<float>& key,
        const xt::xarray<float>& value,
        const xt::xarray<float>* mask = nullptr);

private:
    float dropout_probability;
};