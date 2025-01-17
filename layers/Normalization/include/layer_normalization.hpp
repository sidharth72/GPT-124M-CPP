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
    // Modified constructor to accept weight and bias tensors
    LayerNormalization(const xt::xarray<float>& gamma, const xt::xarray<float>& beta, float eps = 1e-5);
    
    xt::xarray<float> forward(const xt::xarray<float>& x);
    
private:
    xt::xarray<float> weight;  // gamma (stored from input)
    xt::xarray<float> bias;    // beta (stored from input)
    float epsilon;
};