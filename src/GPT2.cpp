#include "tokenizer.hpp"
#include <iostream>
#include <xtensor/xio.hpp> // Add this header for xarray printing
#include "input_embedding.hpp"
#include "layer_normalization.hpp"
#include "multihead_self_attention.hpp"
#include <xtensor/xarray.hpp>
#include <xtensor/xnpy.hpp>
#include "Loader.hpp"

int main() {
    try {
        // Initialize tokenizer with vocabulary file
        GPT2Tokenizer tokenizer("../utils/vocab/gpt2_vocabulary.json");
        // Example text
        std::string text = "Deep learning is one of the revolutionary technologies in the 21st century!";
        
        xt::xarray<int> tokens = tokenizer.encode(text);

        xt::xarray<float> x;
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
        
        InputEmbedding input_embedding(
            token_embed_table, 
            pos_embed_table);

        LayerNormalization layernorm(
            ln1_gamma,
            ln1_beta
        );

        MultiHeadAttention attention(12, 768, 64, 64, 0.1f);



        x = input_embedding.forward(tokens);
        x = layernorm.forward(x);
        x = attention.forward(x, attn_c_attn_weight, attn_c_proj_weight, attn_c_attn_bias, attn_c_proj_bias);

        std::cout << "Layer Norm: " << xt::adapt(x.shape()) << std::endl;

        

    } catch (const std::exception& e) {
        // Print detailed error message
        std::cerr << "Error: " << e.what() << std::endl;
        
        return 1;
    }
    
    return 0;
}