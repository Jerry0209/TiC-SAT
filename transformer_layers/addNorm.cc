//
// Created by alireza on 3/2/22.
//

#include "addNorm.h"
#include <cmath>

AddNormalize::AddNormalize(std::size_t seq_len, std::size_t input_dim,
                           std::size_t kernelDim, std::size_t maxCol) {
    input_dim_ = input_dim;
    seq_len_ = seq_len;
    kernel_dim_ = kernelDim;
    max_col_ = maxCol;
}

void AddNormalize::compute(uint32_t *input, uint32_t *output) {
    for (int i =0; i< seq_len_; i++){
        auto* input_ptr = (int8_t*) (input + i * (input_dim_ >> 2));
        auto* output_ptr = (int8_t*) (output + i * (input_dim_ >> 2));
        int32_t sum = 0;
        for (int j=0; j< input_dim_; j++){
            *output_ptr = (int8_t) (*output_ptr + *input_ptr); // Residual
            sum += *output_ptr;
            output_ptr ++;
            input_ptr ++;
        }

        output_ptr = (int8_t*) (output + i * (input_dim_ >> 2));
        // auto mean = (int32_t) (sum / input_dim_);
        auto mean = (int32_t) (sum / static_cast<int32_t>(input_dim_));
        int32_t variance = 0;
        for (int j=0; j< input_dim_; j++){
            // variance+= (*output_ptr++ - mean) ^ 2; // Assuming that the values are fixed-point with 2 digit of fraction.
            
            /* Fix bug: variance calculation */
            int32_t diff = static_cast<int32_t>(*output_ptr++) - mean;
            variance += diff * diff;
        }
        // variance = variance / (int) input_dim_;
        // double sd = sqrt((double) variance);
        // auto sd_inv = (int32_t) ((1<<2)/(sd + 1)); // prevent zero divide! // Assuming that the values are fixed-point with 2 digit of fraction.

        // // ===== Debug =====
        // std::cout << "row " << i << ", mean = " << mean << std::endl;
        // std::cout << "row " << i
        //           << ", variance = " << variance
        //           << ", sd = " << sd
        //           << ", sd_inv = " << sd_inv
        //           << std::endl;
        // // ====================

        // output_ptr = (int8_t*) (output + i * (input_dim_ >> 2));
        // for (int j=0; j< input_dim_; j++){
        //     *output_ptr = (int8_t) ((*output_ptr - mean) * (sd_inv) >> 2);
        //     output_ptr ++;
        // }

        /* Increase the fixed-point scale for integer addNorm */
        // Doesn't match the output as the notebook LayerNorm
        variance = variance / (int) input_dim_;
        double sd = sqrt((double) variance);
        auto sd_inv = (int32_t)((1 << 8) / (sd + 1));

        output_ptr = (int8_t*) (output + i * (input_dim_ >> 2));
        for (int j=0; j< input_dim_; j++){
            *output_ptr = (int8_t)((((*output_ptr - mean) * sd_inv)) >> 8);
            output_ptr ++;
        }

    }
}

void AddNormalize::computeInterleaved4D(int8_t *input_interleaved, int8_t *output_interleaved) {
    for (std::size_t seq = 0; seq < seq_len_; seq++) {
        int32_t sum[4] = {0, 0, 0, 0};

        for (std::size_t feature = 0; feature < input_dim_; feature++) {
            int8_t* out_slot = output_interleaved + ((seq * input_dim_ + feature) * 4u);
            const int8_t* in_slot = input_interleaved + ((seq * input_dim_ + feature) * 4u);

            for (std::size_t learner = 0; learner < 4u; learner++) {
                out_slot[learner] = static_cast<int8_t>(out_slot[learner] + in_slot[learner]);
                sum[learner] += out_slot[learner];
            }
        }

        int32_t mean[4];
        for (std::size_t learner = 0; learner < 4u; learner++) {
            mean[learner] = sum[learner] / static_cast<int32_t>(input_dim_);
        }

        int32_t variance[4] = {0, 0, 0, 0};
        for (std::size_t feature = 0; feature < input_dim_; feature++) {
            const int8_t* out_slot = output_interleaved + ((seq * input_dim_ + feature) * 4u);
            for (std::size_t learner = 0; learner < 4u; learner++) {
                int32_t diff = static_cast<int32_t>(out_slot[learner]) - mean[learner];
                variance[learner] += diff * diff;
            }
        }

        int32_t sd_inv[4];
        for (std::size_t learner = 0; learner < 4u; learner++) {
            variance[learner] = variance[learner] / static_cast<int32_t>(input_dim_);
            double sd = sqrt(static_cast<double>(variance[learner]));
            sd_inv[learner] = static_cast<int32_t>((1 << 8) / (sd + 1));
        }

        for (std::size_t feature = 0; feature < input_dim_; feature++) {
            int8_t* out_slot = output_interleaved + ((seq * input_dim_ + feature) * 4u);
            for (std::size_t learner = 0; learner < 4u; learner++) {
                out_slot[learner] = static_cast<int8_t>(
                    ((static_cast<int32_t>(out_slot[learner]) - mean[learner]) *
                     sd_inv[learner]) >> 8);
            }
        }
    }
}

void AddNormalize::computeInterleaved2D(int8_t *input_interleaved, int8_t *output_interleaved) {
    for (std::size_t seq = 0; seq < seq_len_; seq++) {
        int32_t sum[2] = {0, 0};

        for (std::size_t feature = 0; feature < input_dim_; feature++) {
            int8_t* out_slot = output_interleaved + ((seq * input_dim_ + feature) * 2u);
            const int8_t* in_slot = input_interleaved + ((seq * input_dim_ + feature) * 2u);

            for (std::size_t learner = 0; learner < 2u; learner++) {
                out_slot[learner] = static_cast<int8_t>(out_slot[learner] + in_slot[learner]);
                sum[learner] += out_slot[learner];
            }
        }

        int32_t mean[2];
        for (std::size_t learner = 0; learner < 2u; learner++) {
            mean[learner] = sum[learner] / static_cast<int32_t>(input_dim_);
        }

        int32_t variance[2] = {0, 0};
        for (std::size_t feature = 0; feature < input_dim_; feature++) {
            const int8_t* out_slot = output_interleaved + ((seq * input_dim_ + feature) * 2u);
            for (std::size_t learner = 0; learner < 2u; learner++) {
                int32_t diff = static_cast<int32_t>(out_slot[learner]) - mean[learner];
                variance[learner] += diff * diff;
            }
        }

        int32_t sd_inv[2];
        for (std::size_t learner = 0; learner < 2u; learner++) {
            variance[learner] = variance[learner] / static_cast<int32_t>(input_dim_);
            double sd = sqrt(static_cast<double>(variance[learner]));
            sd_inv[learner] = static_cast<int32_t>((1 << 8) / (sd + 1));
        }

        for (std::size_t feature = 0; feature < input_dim_; feature++) {
            int8_t* out_slot = output_interleaved + ((seq * input_dim_ + feature) * 2u);
            for (std::size_t learner = 0; learner < 2u; learner++) {
                out_slot[learner] = static_cast<int8_t>(
                    ((static_cast<int32_t>(out_slot[learner]) - mean[learner]) *
                     sd_inv[learner]) >> 8);
            }
        }
    }
}


void AddNormalize::computeRearranged(uint32_t *input, uint32_t *output) {
    auto* input_ptr = (int8_t*) (input );
    auto* output_ptr = (int8_t*) (output);
    for (int i =0; i< seq_len_* input_dim_; i++){
        *output_ptr = (int8_t) (*output_ptr + *input_ptr);
        output_ptr ++;
        input_ptr ++;
    }

    for (int i=0; i< seq_len_; i++){
        output_ptr = ((int8_t*) output) + i*kernel_dim_;
        int sum = 0;
        for (int j =0; j< input_dim_ / kernel_dim_; j++){
            for (int k=0; k< kernel_dim_; k++) {
                sum += *(output_ptr+k);
            }
            output_ptr += seq_len_* kernel_dim_;
        }

        // auto mean = (int32_t) (sum / input_dim_);
        auto mean = (int32_t) (sum / static_cast<int32_t>(input_dim_));
        int32_t variance = 0;
        output_ptr = (int8_t*) output + i*kernel_dim_;
        for (int j =0; j< input_dim_ / kernel_dim_; j++){
            for (int k=0; k< kernel_dim_; k++) {
                // variance+= (*(output_ptr+k) - mean) ^ 2; // Assuming that the values are fixed-point with 2 digit of fraction.
                /* Fix bug: variance calculation */
                int32_t diff = static_cast<int32_t>(*(output_ptr+k)) - mean;
                variance += diff * diff;
            }
            output_ptr += seq_len_* kernel_dim_;
        }

        variance = variance / (int) input_dim_;
        double sd = sqrt((double) variance);
        auto sd_inv = (int32_t) ((1<<2)/(sd + 1)); // prevent zero divide! // Assuming that the values are fixed-point with 2 digit of fraction.

        output_ptr = (int8_t*) output + i*kernel_dim_;
        for (int j =0; j< input_dim_ / kernel_dim_; j++){
            for (int k=0; k< kernel_dim_; k++) {
                *(output_ptr+k) = (int8_t) ((*(output_ptr+k) - mean) * (sd_inv) >> 2);
            }
            output_ptr += seq_len_* kernel_dim_;
        }
    }
}
