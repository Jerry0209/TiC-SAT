#pragma once

// #include <cstddef>
// #include <vector>
// #include <unordered_map>
// #include <string>
#include "util.h"
#include "linearLayer.h"
#include "../accelerator/smm_gem.h"

class Dense : public LinearLayer {
public:
    Dense(std::size_t input_dim, std::size_t output_dim, uint32_t *weight);

    ~Dense() override;

    void compute(std::size_t seq_len, uint32_t *input, uint32_t *output) override;

private:
    void multiplyweight(std::size_t seq_len, uint32_t *input, uint32_t *output);

    void addbias(std::size_t seq_len, uint32_t *output);

    std::size_t input_size_;
    std::size_t output_size_;
    uint32_t *weight; // shape [input_size_, output_size_]
    uint32_t *bias;   // shape [output_size_]
};