#ifndef UTILS_BLOCK_H
#define UTILS_BLOCK_H

#include <stdexcept>
#include <string>

#include "scran_blocks/scran_blocks.hpp"

#include "utils.h"

inline scran_blocks::WeightPolicy parse_block_weight_policy(const std::string& block_weight_policy) {
    scran_blocks::WeightPolicy output  = scran_blocks::WeightPolicy::NONE;
    if (block_weight_policy == "none") {
    } else if (block_weight_policy == "equal") {
        output = scran_blocks::WeightPolicy::EQUAL;
    } else if (block_weight_policy == "variable") {
        output = scran_blocks::WeightPolicy::VARIABLE;
    } else {
        throw std::runtime_error("unknown block weight policy '" + block_weight_policy + "'");
    }
    return output;
}

inline scran_blocks::VariableWeightParameters parse_variable_block_weight(const pybind11::tuple& variable_block_weight) {
    if (variable_block_weight.size() != 2) {
        throw std::runtime_error("'variable_block_weight' must be a numeric vector of length 2");
    }

    scran_blocks::VariableWeightParameters output;
    output.lower_bound = variable_block_weight[0].cast<double>();
    output.upper_bound = variable_block_weight[1].cast<double>();
    return output;
}

#endif
