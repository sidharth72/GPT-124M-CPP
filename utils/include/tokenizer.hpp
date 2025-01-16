#pragma once

#include <string>
#include <vector>
#include <map>
#include <regex>
#include <nlohmann/json.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xadapt.hpp>
#include <algorithm>

class GPT2Tokenizer {
public:
    GPT2Tokenizer(const std::string& vocab_path);
    
    // Main tokenization methods
    xt::xarray<int> encode(const std::string& text);
    std::string decode(const xt::xarray<int>& tokens);
    
private:
    // Token mappings
    std::map<std::string, int> encoder; // maps string token to integer id
    std::map<int, std::string> decoder; // maps integer id to string token
    
    // Byte level encodings
    std::map<unsigned char, char> byte_encoder;
    std::map<char, unsigned char> byte_decoder;
    
    // Helper methods
    void initialize_byte_encodings();
    std::string byte_encode(const std::string& text);
    std::string byte_decode(const std::string& text);
    
    // Regex pattern for tokenization
    std::regex pat;
};