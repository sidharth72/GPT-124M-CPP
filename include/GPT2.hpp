#pragma once
#include "tokenizer.hpp"
#include "input_embedding.hpp"
#include "layer_normalization.hpp"
#include "multihead_self_attention.hpp"
#include "mlp.hpp"
#include "Loader.hpp"
#include "activations.hpp"
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xmanipulation.hpp>
#include <xtensor/xmath.hpp>  // For argmax
#include <xtensor/xsort.hpp>  // For argsort
#include <string>
#include <memory>  // For smart pointers
#include <random>

class GPT2 {
public:
    struct Config {
        size_t num_layers;
        size_t num_heads;
        size_t d_model;
        size_t d_k;
        size_t d_v;
        size_t d_ff;
        size_t vocab_size;
        float dropout_rate;
    };

    GPT2(const std::string& model_path, const std::string& vocab_path) 
        : tokenizer(vocab_path),
          config{12, 12, 768, 64, 64, 3072, 50257, 0.0},
          mha(config.num_heads, config.d_model, config.d_k, config.d_v) {  // Initialize MHA with parameters
        initialize(model_path);
    }

    // Make destructor virtual and public
    virtual ~GPT2() = default;
    std::string generate_next_token(const std::string& input_text, int k) {
        // Tokenize input
        xt::xarray<int> tokens = tokenizer.encode(input_text);
        
        // Create look ahead mask
        xt::xarray<float> look_ahead_mask = create_look_ahead_mask(tokens.shape()[0]);
        
        // Forward pass
        auto logits = forward(tokens, look_ahead_mask);
        
        // Get last token probabilities
        xt::xarray<float> last_token_probs = xt::view(logits, xt::all(), -1, xt::all());

        //Flatten the probabilities
        xt::xarray<float> flat_probs = xt::flatten(last_token_probs);
        xt::xarray<size_t> sorted_indices = xt::argsort(flat_probs);
        sorted_indices = xt::flip(sorted_indices);

        // Get top k indices and their probabilities
        xt::xarray<size_t> top_k_indices = xt::view(sorted_indices, xt::range(0, k));
        xt::xarray<float> top_k_probs = xt::zeros<float>({k});

        // Extract probabilities for top k indices
        for(size_t i = 0; i < k; ++i) {
            top_k_probs[i] = flat_probs[top_k_indices[i]];
        }
        
        // Normalize the top k probabilities
        float sum = xt::sum(top_k_probs)();
        top_k_probs = top_k_probs / sum;
        
        // Random sampling
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0, 1.0);
        
        float rand_val = dis(gen);
        float cumsum = 0.0;
        
        // Sample based on normalized probabilities
        for(size_t i = 0; i < k; ++i) {
            cumsum += top_k_probs[i];
            if(rand_val <= cumsum) {
                // Create a single-element xarray for the token ID
                xt::xarray<int> token_id = {static_cast<int>(top_k_indices[i])};
                return tokenizer.decode(token_id);
            }
        }

        // Fallback case: decode the most probable token
        xt::xarray<int> token_id = {static_cast<int>(top_k_indices[0])};
        return tokenizer.decode(token_id);
    }

private:
    Config config;
    GPT2Tokenizer tokenizer;
    std::unique_ptr<InputEmbedding> input_embedding;  // Use smart pointer
    LayerNormalization layernorm;
    MultiHeadAttention mha;
    MLP mlp;
    
    // Store model parameters
    std::unordered_map<std::string, xt::xarray<float>> parameters;
    
    void initialize(const std::string& model_path) {
        // Load weights
        GPT2WeightLoader loader;
        parameters = loader.loadWeights(model_path);
        
        // Initialize embedding layers using smart pointer
        input_embedding = std::make_unique<InputEmbedding>(
            parameters["transformer.wte.weight"],
            parameters["transformer.wpe.weight"]
        );
    }
    
    xt::xarray<float> create_look_ahead_mask(size_t seq_length) {
        xt::xarray<float> mask = xt::triu(xt::ones<float>({seq_length, seq_length}), 1);
        return xt::expand_dims(mask, 0);
    }
    
    xt::xarray<float> forward(const xt::xarray<int>& tokens, const xt::xarray<float>& look_ahead_mask) {
        std::string path_prefix = "transformer.h.";
        
        // Input embedding
        auto x = input_embedding->forward(tokens);
        
        // Transform through layers
        for(size_t i = 0; i < config.num_layers; ++i) {
            std::string layer_prefix = path_prefix + std::to_string(i) + ".";
            
            // Layer normalization 1
            auto ln1_out = layernorm.forward(
                x,
                parameters[layer_prefix + "ln_1.weight"],
                parameters[layer_prefix + "ln_1.bias"]
            );
            
            // Self attention
            auto attn_out = mha.forward(
                ln1_out,
                parameters[layer_prefix + "attn.c_attn.weight"],
                parameters[layer_prefix + "attn.c_proj.weight"],
                parameters[layer_prefix + "attn.c_attn.bias"],
                parameters[layer_prefix + "attn.c_proj.bias"],
                &look_ahead_mask
            );
            
            x = x + attn_out;
            
            // Layer normalization 2
            auto ln2_out = layernorm.forward(
                x,
                parameters[layer_prefix + "ln_2.weight"],
                parameters[layer_prefix + "ln_2.bias"]
            );
            
            // MLP
            auto mlp_out = mlp.forward(
                ln2_out,
                parameters[layer_prefix + "mlp.c_fc.weight"],
                parameters[layer_prefix + "mlp.c_fc.bias"],
                parameters[layer_prefix + "mlp.c_proj.weight"],
                parameters[layer_prefix + "mlp.c_proj.bias"]
            );
            
            x = x + mlp_out;
        }
        
        // Final layer norm
        x = layernorm.forward(
            x,
            parameters["transformer.ln_f.weight"],
            parameters["transformer.ln_f.bias"]
        );
        
        // Output projection and softmax
        auto logits = xt::linalg::dot(x, xt::transpose(parameters["lm_head.weight"]));
        return activation::Softmax::forward(logits, 2);
    }
};