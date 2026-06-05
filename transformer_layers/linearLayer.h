#pragma once

#include <cstddef>
#include <cstdint>

class LinearLayer {
public:
    virtual ~LinearLayer() = default;

    /**
     * Compute one linear layer over a sequence of input rows.
     *
     * @param seq_len Number of sequence positions, or rows, to process. Each
     *        row has the layer-specific input feature width.
     * @param input Pointer to the packed input activation matrix. The concrete
     *        layer implementation defines the feature width, but the common
     *        Dense/CodebookDense path stores four signed 8-bit values in each
     *        uint32_t word.
     * @param output Pointer to the packed output activation matrix. The caller
     *        owns the buffer and must allocate enough words for seq_len rows
     *        and the layer-specific output feature width.
     */
    virtual void compute(std::size_t seq_len, uint32_t *input, uint32_t *output) = 0;
};
