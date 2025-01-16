#include "tokenizer.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <regex>

GPT2Tokenizer::GPT2Tokenizer(const std::string& vocab_path) {
    std::ifstream vocab_file(vocab_path);
    if (!vocab_file.is_open()) {
        throw std::runtime_error("Failed to open vocabulary file");
    }
    
    nlohmann::json vocab_json;
    vocab_file >> vocab_json;
    
    for (auto& [token, id] : vocab_json["token_to_id"].items()) {
        encoder[token] = id.get<int>();
        decoder[id.get<int>()] = token;
    }
    
    initialize_byte_encodings();
}

// Initialize byte level encodings for GPT2 tokenizer, this is used to handle special characters

void GPT2Tokenizer::initialize_byte_encodings() {
    std::vector<int> bytes_to_unicode_list;
    
    // Initialize basic ASCII ranges
    for (int i = '!'; i <= '~'; i++) bytes_to_unicode_list.push_back(i);
    for (int i = '¡'; i <= '¬'; i++) bytes_to_unicode_list.push_back(i);
    for (int i = '®'; i <= 'ÿ'; i++) bytes_to_unicode_list.push_back(i);
    
    std::vector<int> n_bytes_to_unicode_list;
    int n = 0;
    
    // Fill in missing bytes
    for (int b = 0; b < 256; b++) {
        if (std::find(bytes_to_unicode_list.begin(), bytes_to_unicode_list.end(), b) 
            == bytes_to_unicode_list.end()) {
            bytes_to_unicode_list.push_back(b);
            n_bytes_to_unicode_list.push_back(256 + n);
            n++;
        } else {
            n_bytes_to_unicode_list.push_back(b);
        }
    }
    
    // Create bidirectional mappings
    for (size_t i = 0; i < 256; i++) {
        char original_byte = static_cast<char>(i);
        char unicode_char = static_cast<char>(n_bytes_to_unicode_list[i]);
        byte_encoder[original_byte] = unicode_char;
        byte_decoder[unicode_char] = original_byte;
    }
}

// Encode text to byte level encodings

std::string GPT2Tokenizer::byte_encode(const std::string& text) {
    std::string result;
    result.reserve(text.length());
    
    for (unsigned char c : text) {
        auto it = byte_encoder.find(c);
        if (it != byte_encoder.end()) {
            result += it->second;
        }
    }
    return result;
}


// Decode byte level encodings to text

std::string GPT2Tokenizer::byte_decode(const std::string& text) {
    std::string result;
    result.reserve(text.length());
    
    for (char c : text) {
        auto it = byte_decoder.find(c);
        if (it != byte_decoder.end()) {
            result += it->second;
        }
    }
    return result;
}


// encode text to integer tokens

xt::xarray<int> GPT2Tokenizer::encode(const std::string& text) {
    std::vector<int> bpe_tokens;
    std::string byte_encoded = byte_encode(text);
    
    size_t i = 0;
    bool prev_is_whitespace = false;  // start with false since we don't need to consider whitespace at the beginning
    

    // A sliding window approach to find the longest token possible
    while (i < byte_encoded.length()) {
        bool found_token = false;
        std::string prefix = prev_is_whitespace ? "Ġ" : "";
        
        // Try to match the longest token possible
        for (size_t len = std::min<size_t>(100, byte_encoded.length() - i); len > 0; len--) {
            std::string substr = byte_encoded.substr(i, len);
            std::string token_to_find = prefix + substr;
            
            auto it = encoder.find(token_to_find);
            if (it != encoder.end()) {
                bpe_tokens.push_back(it->second);
                i += len;
                found_token = true;
                prev_is_whitespace = false;
                break;
            }
        }
        
        if (!found_token) {
            // Handle single character if no token match found
            char c = byte_encoded[i];
            prev_is_whitespace = std::isspace(static_cast<unsigned char>(c));
            std::string chr(1, c);
            auto it = encoder.find(chr);
            if (it != encoder.end()) {
                bpe_tokens.push_back(it->second);
            }
            i++;
        }
    }
    
    return xt::adapt(bpe_tokens);
}

std::string GPT2Tokenizer::decode(const xt::xarray<int>& tokens) {
    std::string text;
    for (int token : tokens) {
        auto it = decoder.find(token);
        if (it != decoder.end()) {
            text += it->second;
        }
    }
    
    // Remove the special 'Ġ' character that represents spaces
    std::string processed_text;
    for (size_t i = 0; i < text.length(); i++) {
        if (text[i] == 'Ġ') {
            processed_text += ' ';
        } else {
            processed_text += text[i];
        }
    }

    // Remove the unnecessary pattern
    // EG: There─áis─ásomething─áthat─áwe─ádon't─áknow.─áBecause─áwe─ádon't─áknow─áit.
    // We need to remove the unwanted characters like 
    std::regex pattern("Ġ"); // Ġ is the pattern that fills the spaces, so we need to remove it
    processed_text = std::regex_replace(processed_text, pattern, " ");

    
    // encode first and then decode byte level encodings
    return processed_text;
}