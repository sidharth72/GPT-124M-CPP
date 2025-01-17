// #include "tokenizer.hpp"
// #include <iostream>
// #include <xtensor/xio.hpp> // Add this header for xarray printing
// #include "input_embedding.hpp"
// #include "layer_normalization.hpp"
// #include <xtensor/xarray.hpp>
// #include <xtensor/xnpy.hpp>

// int main() {
//     try {
//         // Initialize tokenizer with vocabulary file
//         GPT2Tokenizer tokenizer("../utils/vocab/gpt2_vocabulary.json");
  
//         // Example text
//         std::string text = "Deep learning is one of the revolutionary technologies in the 21st century!";
        
//         // Encode text to tokens
//         xt::xarray<int> tokens = tokenizer.encode(text);

//         // Initialize token embedding table and positional embedding table
//         xt::xarray<float> token_embed_table = xt::load_npy<float>("../parameters/gpt2/transformer.wte.weight.npy");
//         xt::xarray<float> pos_embed_table = xt::load_npy<float>("../parameters/gpt2/transformer.wpe.weight.npy");


//         // Initialize input embedding layer

//         InputEmbedding input_embedding(token_embed_table, pos_embed_table);

//         std::string gamma_path = "../parameters/gpt2/transformer.h.0.ln_1.weight.npy";
//         std::string beta_path = "../parameters/gpt2/transformer.h.0.ln_1.bias.npy";

//         xt::xarray<float> gamma = xt::load_npy<float>(gamma_path);
//         xt::xarray<float> beta = xt::load_npy<float>(beta_path);

//         LayerNormalization layernorm(gamma, beta);
        

//         // Get input embeddings
//         xt::xarray<float> input_embeddings = input_embedding.forward(tokens);
//         xt::xarray<float> layernorm_output = layernorm.forward(input_embeddings);

//         std::cout << "Layer Norm: " << layernorm_output << std::endl;

//         // Decode tokens back to text
//         std::string decoded_text = tokenizer.decode(tokens);
        
//         // Print results
//         std::cout << "Original text: " << text << std::endl;
//         std::cout << "Tokens: " << tokens << std::endl;
//         std::cout << "Decoded text: " << decoded_text << std::endl;
        
//     } catch (const std::exception& e) {
//         std::cerr << "Error: " << e.what() << std::endl;
//         return 1;
//     }
    
//     return 0;
// }