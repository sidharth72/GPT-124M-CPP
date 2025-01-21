#include "tokenizer.hpp"
#include <iostream>
#include <xtensor/xio.hpp> // Add this header for xarray printing
#include "input_embedding.hpp"
#include "layer_normalization.hpp"
#include "multihead_self_attention.hpp"
#include <xtensor/xarray.hpp>
#include <xtensor/xnpy.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xreducer.hpp>
#include "Loader.hpp"


xt::xarray<float> create_look_ahead_mask(size_t seq_length) {
    // Create a square matrix of shape [seq_length, seq_length]
    xt::xarray<float> mask = xt::triu(xt::ones<float>({seq_length, seq_length}), 1);
    
    // Convert to attention mask format where:
    // - 0.0 indicates positions to attend to
    // - 1.0 indicates positions to mask out
    
    // Expand dimensions to include batch size dimension [1, seq_length, seq_length]
    std::vector<std::size_t> expanded_shape = {1, seq_length, seq_length};
    xt::xarray<float> expanded_mask = xt::expand_dims(mask, 0);
    
    return expanded_mask;
}

int main() {
    try {
        // Initialize tokenizer with vocabulary file
        GPT2Tokenizer tokenizer("../utils/vocab/gpt2_vocabulary.json");
        // Example text
        std::string text = "Deep learning is one of the revolutionary technologies in the 21st century!";
        
        xt::xarray<int> tokens = tokenizer.encode(text);

        xt::xarray<float> x;
        xt::xarray<float>ln1_out;
        xt::xarray<float>ln2_out;

        GPT2WeightLoader loader;
        auto parameters = loader.loadWeights("../parameters/gpt2");

        // Initialize input embedding layer, token embedding table and positional embedding table
        xt::xarray<float> token_embed_table = parameters["transformer.wte.weight"];
        xt::xarray<float> pos_embed_table = parameters["transformer.wpe.weight"];
        xt::xarray<float> ln1_gamma = parameters["transformer.h.0.ln_1.weight"];
        xt::xarray<float> ln1_beta = parameters["transformer.h.0.ln_1.bias"];
        xt::xarray<float> attn_c_attn_weight = parameters["transformer.h.0.attn.c_attn.weight"];
        xt::xarray<float> attn_c_proj_weight = parameters["transformer.h.0.attn.c_proj.weight"];
        xt::xarray<float> attn_c_attn_bias = parameters["transformer.h.0.attn.c_attn.bias"];
        xt::xarray<float> attn_c_proj_bias = parameters["transformer.h.0.attn.c_proj.bias"];
        xt::xarray<float> ln2_gamma = parameters["transformer.h.0.ln_2.weight"];
        xt::xarray<float> ln2_beta = parameters["transformer.h.0.ln_2.bias"];

        
        InputEmbedding input_embedding(
            token_embed_table, 
            pos_embed_table);

        LayerNormalization layernorm;
        MultiHeadAttention attention(12, 768, 64, 64, 0.1f);

        // Create a look ahead mask of shape [1, seq_len, seq_len]
        xt::xarray<float> look_ahead_mask = create_look_ahead_mask(tokens.shape()[0]);

        std::cout << "mask: " << look_ahead_mask << std::endl;


        x = input_embedding.forward(tokens);
        ln1_out = layernorm.forward(x, ln1_gamma, ln1_beta);
        x = attention.forward(ln1_out, attn_c_attn_weight, attn_c_proj_weight, attn_c_attn_bias, attn_c_proj_bias, &look_ahead_mask);
        x = x + ln1_out;
        ln2_out = layernorm.forward(x, ln2_gamma, ln2_beta);

        std::cout << "Layer norm2 output: " << ln2_out << std::endl;  

        

    } catch (const std::exception& e) {
        // Print detailed error message
        std::cerr << "Error: " << e.what() << std::endl;
        
        return 1;
    }
    
    return 0;
}