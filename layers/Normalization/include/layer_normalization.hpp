#pragma once
#include "xtensor/xbuilder.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xadapt.hpp"
#include "xtensor/xio.hpp"
#include <xtensor/xarray.hpp>
#include <xtensor/xnpy.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xreducer.hpp>

class LayerNormalization {
public:
    // Constructor now only takes epsilon
    explicit LayerNormalization(float eps = 1e-05);
    
    // Forward pass now takes weights and biases as parameters
    xt::xarray<float> forward(
        const xt::xarray<float>& x,
        const xt::xarray<float>& gamma,
        const xt::xarray<float>& beta
    );
    
private:
    float epsilon;
};