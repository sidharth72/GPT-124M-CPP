#include "Loader.hpp"
#include <xtensor/xnpy.hpp>
#include <filesystem>

GPT2WeightLoader::GPT2WeightLoader() {
    initializeWeightPaths();
}

void GPT2WeightLoader::initializeWeightPaths() {
    // Embeddings
    weight_paths.push_back("transformer.wte.weight");
    weight_paths.push_back("transformer.wpe.weight");
    
    // Transformer blocks
    for (int layer = 0; layer < 12; layer++) {
        std::string prefix = "transformer.h." + std::to_string(layer);
        
        // Layer norms
        weight_paths.push_back(prefix + ".ln_1.weight");
        weight_paths.push_back(prefix + ".ln_1.bias");
        weight_paths.push_back(prefix + ".ln_2.weight");
        weight_paths.push_back(prefix + ".ln_2.bias");
        
        // Attention
        weight_paths.push_back(prefix + ".attn.c_attn.weight");
        weight_paths.push_back(prefix + ".attn.c_attn.bias");
        weight_paths.push_back(prefix + ".attn.c_proj.weight");
        weight_paths.push_back(prefix + ".attn.c_proj.bias");
        
        // MLP
        weight_paths.push_back(prefix + ".mlp.c_fc.weight");
        weight_paths.push_back(prefix + ".mlp.c_fc.bias");
        weight_paths.push_back(prefix + ".mlp.c_proj.weight");
        weight_paths.push_back(prefix + ".mlp.c_proj.bias");
    }
    
    // Final layer norm and output
    weight_paths.push_back("transformer.ln_f.weight");
    weight_paths.push_back("transformer.ln_f.bias");
    weight_paths.push_back("lm_head.weight");
}

GPT2WeightLoader::WeightMap GPT2WeightLoader::loadWeights(const std::string& weight_dir) {
    WeightMap weights;
    
    for (const auto& path : weight_paths) {
        std::string full_path = weight_dir + "/" + path + ".npy";
        try {
            weights[path] = xt::load_npy<float>(full_path);
        } catch (const std::exception& e) {
            std::cerr << "Error loading weight file " << full_path << ": " << e.what() << std::endl;
            throw;
        }
    }
    
    return weights;
}

const std::vector<std::string>& GPT2WeightLoader::getWeightPaths() const {
    return weight_paths;
}