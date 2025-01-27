// main.cpp
#include "GPT2.hpp"
#include <iostream>
#include <xtensor/xsort.hpp>  // For argsort

// Modified main.cpp
int main() {
    try {
        GPT2 model("../parameters/gpt2", "../utils/vocab/gpt2_vocabulary.json");
        std::string text = "Once there is a man named";
        std::cout << "Initial prompt: " << text << std::endl;

        GPT2Tokenizer tokenizer("../utils/vocab/gpt2_vocabulary.json");
        
        const int max_tokens = 15;
        const int k = 5;  // Top-5 sampling
        
        for(int i = 0; i < max_tokens; i++) {
            // Generate next token probabilities
            std::string next_token = model.generate_next_token(text, k);

            text += next_token;

            // std::cout << next_token << std::flush;
            
            
            // Optional: Add stopping condition for newline
            if(next_token == "\n") {
                break;
            }
        }
        
        std::cout << "Generated text: " << text << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}