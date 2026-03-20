#pragma once

#include <cstddef>
#include <cstdint>

class LinearLayer {
public:
    virtual ~LinearLayer() = default;
    virtual void compute(std::size_t seq_len, uint32_t *input, uint32_t *output) = 0;
};
