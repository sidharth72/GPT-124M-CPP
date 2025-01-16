#include "tokenizer.hpp"
#include <iostream>
#include <xtensor/xio.hpp> // Add this header for xarray printing
#include "input_embedding.hpp"
#include <xtensor/xarray.hpp>
#include <xtensor/xnpy.hpp>

int main() {
    try {
        // Initialize tokenizer with vocabulary file
        GPT2Tokenizer tokenizer("../utils/vocab/gpt2_vocabulary.json");
        
        // Example text
        std::string text = "Deep Learning is one of the revolutionary technologies in the field of computer science.";
        
        // Encode text to tokens
        xt::xarray<int> tokens = tokenizer.encode(text);

        // Initialize token embedding table and positional embedding table
        xt::xarray<float> token_embed_table = xt::load_npy<float>("../parameters/gpt2/transformer.wte.weight.npy");
        xt::xarray<float> pos_embed_table = xt::load_npy<float>("../parameters/gpt2/transformer.wpe.weight.npy");


        // Initialize input embedding layer

        InputEmbedding input_embedding(token_embed_table, pos_embed_table);

        // Get input embeddings
        xt::xarray<float> input_embeddings = input_embedding.forward(tokens);

        // Print the shape of the input embedding
        std::cout << xt::adapt(input_embeddings.shape()) << std::endl;


        std::cout << token_embed_table << std::endl;
        // Decode tokens back to text
        std::string decoded_text = tokenizer.decode(tokens);
        
        // Print results
        std::cout << "Original text: " << text << std::endl;
        std::cout << "Tokens: " << tokens << std::endl;
        std::cout << "Decoded text: " << decoded_text << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}