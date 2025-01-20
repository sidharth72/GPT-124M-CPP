#pragma once
#include <xtensor/xarray.hpp>
#include <string>
#include <unordered_map>
#include <vector>

class GPT2WeightLoader {
public:
    using WeightMap = std::unordered_map<std::string, xt::xarray<float>>;
    
    GPT2WeightLoader();
    
    // Load all weights from the given directory path
    WeightMap loadWeights(const std::string& weight_dir);
    
    // Get list of all weight paths for reference
    const std::vector<std::string>& getWeightPaths() const;

private:
    std::vector<std::string> weight_paths;
    void initializeWeightPaths();
};