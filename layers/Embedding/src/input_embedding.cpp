/*

The code three arguments, the look up table for token embedding and positional embeddings, and the input sequence of integer tokens.
Then embeddings are taken from the lookup table based on the token indices.
Then the token embeddings is being concatenated with the positional embeddings to get the input embedding to the transformer.

*/

#include "input_embedding.hpp"
#include <xtensor/xbuilder.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xadapt.hpp>


// Constructor takes the token embedding table and positional embedding table
InputEmbedding::InputEmbedding(const xt::xarray<float>& token_embed_table,
                               const xt::xarray<float>& pos_embed_table)
    : token_embeddings(token_embed_table), positional_embeddings(pos_embed_table) {

    }

xt::xarray<float> InputEmbedding::forward(const xt::xarray<int>& input_tokens) {

    std::size_t batch_size = 1; // Batch size is probably 1 for inference
    std::size_t seq_length = input_tokens.shape()[0];
    std::size_t embed_dim = token_embeddings.shape()[1];

    // The output tensor, shape: (batch_size, seq_length, embed_dim)
    xt::xarray<float> output = xt::zeros<float>({batch_size, seq_length, embed_dim});

    for (std::size_t b = 0; b < batch_size; b ++){

        // Itereate over the input tokens
        for(std::size_t i = 0; i < seq_length; i++) {

            // Get the token indices
            auto token_idx = input_tokens(i);

            // Get the token embeddings and positional embeddings
            auto token_embed = xt::view(token_embeddings, token_idx, xt::all());
            auto pos_embed = xt::view(positional_embeddings, i, xt::all());

            // Concate the token embeddings and positional embeddings
            xt::view(output, b, i, xt::all()) = token_embed + pos_embed;

        }

    }

    return output;
}