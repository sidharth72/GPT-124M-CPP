#include "tokenizer.hpp"
#include <iostream>
#include <xtensor/xio.hpp> // Add this header for xarray printing

int main() {
    try {
        // Initialize tokenizer with vocabulary file
        GPT2Tokenizer tokenizer("../utils/vocab/gpt2_vocabulary.json");
        
        // Example text
        std::string text = "Hello, world! This is a test.";
        
        // Encode text to tokens
        xt::xarray<int> tokens = tokenizer.encode(text);
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