#include "tokenizer.hpp"
#include <iostream>
#include <xtensor/xio.hpp> // Add this header for xarray printing
#include "input_embedding.hpp"
#include "layer_normalization.hpp"
#include "multihead_self_attention.hpp"
#include "mlp.hpp"
#include "activations.hpp"
#include <xtensor/xarray.hpp>
#include <xtensor/xnpy.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xreducer.hpp>
#include "Loader.hpp"


// Create a Structure for the GPT configuration and store the config information
struct GPTConfig {
    size_t num_layers;
    size_t num_heads;
    size_t d_model;
    size_t d_k;
    size_t d_v;
    size_t d_ff;
    size_t vocab_size;
    float dropout_rate;
};

// Implemente the GPT config model
GPTConfig gpt_config = {
    12, // Number of layers
    12, // Number of heads
    768, // Model dimension
    64, // Key dimension
    64, // Value dimension
    3072, // Feed forward dimension
    50257, // Vocabulary size
    0.0 // Dropout rate
};


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
        std::string text = "A neural network is a network of";
        
        xt::xarray<int> tokens = tokenizer.encode(text);

        xt::xarray<float> x;
        xt::xarray<float>ln1_out;
        xt::xarray<float>ln2_out;
        xt::xarray<float>attn_out;
        xt::xarray<float>mlp_out;



        GPT2WeightLoader loader;
        auto parameters = loader.loadWeights("../parameters/gpt2");

        // Initialize input embedding layer, token embedding table and positional embedding table
        xt::xarray<float> token_embed_table = parameters["transformer.wte.weight"];
        xt::xarray<float> pos_embed_table = parameters["transformer.wpe.weight"];


        std::string path_prefix = "transformer.h.";

        
        InputEmbedding input_embedding(
            token_embed_table, 
            pos_embed_table);

        LayerNormalization layernorm;

        MultiHeadAttention mha(
            gpt_config.num_heads, 
            gpt_config.d_model, 
            gpt_config.d_k, 
            gpt_config.d_v);

        MLP mlp;


        // Create a look ahead mask of shape [1, seq_len, seq_len]
        xt::xarray<float> look_ahead_mask = create_look_ahead_mask(tokens.shape()[0]);


        // Forward pass through the model

        x = input_embedding.forward(tokens);


        for(size_t i = 0; i < gpt_config.num_layers; ++i) {

            std::string layer_prefix = path_prefix + std::to_string(i) + ".";
            xt::xarray<float> ln1_gamma = parameters[layer_prefix + "ln_1.weight"];
            xt::xarray<float> ln1_beta = parameters[layer_prefix + "ln_1.bias"];
            xt::xarray<float> attn_c_attn_weight = parameters[layer_prefix + "attn.c_attn.weight"];
            xt::xarray<float> attn_c_proj_weight = parameters[layer_prefix + "attn.c_proj.weight"];
            xt::xarray<float> attn_c_attn_bias = parameters[layer_prefix + "attn.c_attn.bias"];
            xt::xarray<float> attn_c_proj_bias = parameters[layer_prefix + "attn.c_proj.bias"];
            xt::xarray<float> ln2_gamma = parameters[layer_prefix + "ln_2.weight"];
            xt::xarray<float> ln2_beta = parameters[layer_prefix + "ln_2.bias"];
            xt::xarray<float> mlp_c_fc_weight = parameters[layer_prefix + "mlp.c_fc.weight"];
            xt::xarray<float> mlp_c_fc_bias = parameters[layer_prefix + "mlp.c_fc.bias"];
            xt::xarray<float> mlp_c_proj_weight = parameters[layer_prefix + "mlp.c_proj.weight"];
            xt::xarray<float> mlp_c_proj_bias = parameters[layer_prefix + "mlp.c_proj.bias"];

            ln1_out = layernorm.forward(x, ln1_gamma, ln1_beta);
            attn_out = mha.forward(ln1_out, attn_c_attn_weight, attn_c_proj_weight, attn_c_attn_bias, attn_c_proj_bias, &look_ahead_mask);
            x = x + attn_out;
            ln2_out = layernorm.forward(x, ln2_gamma, ln2_beta);
            mlp_out = mlp.forward(ln2_out, mlp_c_fc_weight, mlp_c_fc_bias, mlp_c_proj_weight, mlp_c_proj_bias);
            x = x + mlp_out;

        }
        x = layernorm.forward(x, 
                            parameters["transformer.ln_f.weight"], 
                            parameters["transformer.ln_f.bias"]);

        // Get the output projection weight (should be [vocab_size, d_model])
        xt::xarray<float> output_proj_weight = parameters["lm_head.weight"];
        xt::xarray<float> logits = xt::linalg::dot(x, xt::transpose(output_proj_weight));
        xt::xarray<float> probs = activation::Softmax::forward(logits, 2);

        // Take the last token probabilities
        xt::xarray<float> last_token_probs = xt::view(probs, xt::all(), -1, xt::all());
        
        // Get the token id with the highest probability
        int predicted_token_id = xt::argmax(last_token_probs)();

        // Decode the predicted token
        std::string predicted_token = tokenizer.decode(predicted_token_id);

        // Print the predicted token
        std::cout << "Predicted token: " << predicted_token << std::endl;


    } catch (const std::exception& e) {
        // Print detailed error message
        std::cerr << "Error: " << e.what() << std::endl;
        
        return 1;
    }
    
    return 0;
}