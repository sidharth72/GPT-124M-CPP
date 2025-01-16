// input_embedding.hpp
#pragma once
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>

class InputEmbedding {
public:
    InputEmbedding(const xt::xarray<float>& token_embed_table,
                   const xt::xarray<float>& pos_embed_table);
    
    xt::xarray<float> forward(const xt::xarray<int>& input_tokens);

private:
    xt::xarray<float> token_embeddings;
    xt::xarray<float> positional_embeddings;
};