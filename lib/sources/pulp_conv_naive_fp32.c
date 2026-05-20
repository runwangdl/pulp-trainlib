/*
 * Copyright (C) 2021-2022 ETH Zurich and University of Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Authors: Davide Nadalini, Leonardo Ravaglia, Calin Diaconu
*/

#include "pulp_train_utils_fp32.h"
#include "pulp_matmul_fp32.h"
#include "pulp_conv_naive_fp32.h"

#include "pmsis.h"


/** DEPTH-WISE CONVOLUTION KERNELS **/

// Naive forward kernel for DepthWise Convolution
void dw_kernel_forward(void *kernel_DW_args) {
    // Extract arguments
    struct kernel_DW_args *args = (struct kernel_DW_args *) kernel_DW_args;

    float *inData = args->input->data;
    float *coeffData = args->weights->data;
    float *outData = args->output->data;

    uint32_t C_in = args->input->C;
    uint32_t H_in = args->input->H;
    uint32_t W_in = args->input->W;

    uint32_t pH = args->weights->H;
    uint32_t pW = args->weights->W;

    uint32_t H_out = args->output->H;
    uint32_t W_out = args->output->W;

    uint32_t h_str = args->stride_h;
    uint32_t w_str = args->stride_w;

    uint32_t Lpad = args->Lpad;
    uint32_t Rpad = args->Rpad;
    uint32_t Upad = args->Upad;
    uint32_t Dpad = args->Dpad;

    // Split work
    uint32_t blockSize = (C_in + NUM_CORES - 1) / NUM_CORES;
    uint32_t start = pi_core_id() * blockSize;
    uint32_t stop = start + blockSize > C_in ? C_in : start + blockSize;

    // Compute
    if (Lpad + Rpad + Upad + Dpad == 0) {
        // No padding
        for (int ch = start; ch < stop; ch++) {
            for (int ho = 0; ho < H_out; ho++) {
                for (int wo = 0; wo < W_out; wo++) {
                    float temp = 0;

                    for (int hk = 0; hk < pH; hk++) {
                        for (int wk = 0; wk < pW; wk++) {
                            temp += coeffData[wk + hk * pW + ch * pH * pW] *
                                    inData[wo + wk + (ho + hk) * W_in + ch * H_in * W_in];
                        }
                    }

                    outData[wo + ho * W_out + ch * H_out * W_out] = temp;
                }
            }
        }
    } else {
        // Padding
        for (int ch = start; ch < stop; ch++) {
            for (int ho = 0; ho < H_out; ho++) {
                for (int wo = 0; wo < W_out; wo++) {
                    float temp = 0;

                    for (int hk = 0; hk < pH; hk++) {
                        for (int wk = 0; wk < pW; wk++) {
                            int pad_cond_h = h_str * ho + hk - Upad;
                            int pad_cond_w = w_str * wo + wk - Lpad;

                            if (
                                    (pad_cond_h >= 0) &&
                                    (pad_cond_w >= 0) &&
                                    (pad_cond_h < H_in) &&
                                    (pad_cond_w < W_in)) {

                                temp += coeffData[wk + hk * pW + ch * pH * pW] *
                                        inData[wo * w_str + wk - Lpad + (h_str * ho + hk - Upad) * W_in +
                                               ch * H_in * W_in];
                            }
                        }
                    }

                    outData[wo + ho * W_out + ch * H_out * W_out] = temp;
                }
            }
        }
    }
}


// Naive weight grad kernel for DepthWise Convolution (stride=1, no padding)
void dw_kernel_weight_grad(void *kernel_DW_args) {
    struct kernel_DW_args *args = (struct kernel_DW_args *) kernel_DW_args;

    float *inData = args->input->data;
    float *coeffDiff = args->weights->diff;
    float *outDiff = args->output->diff;

    uint32_t C_in = args->input->C;
    uint32_t H_in = args->input->H;
    uint32_t W_in = args->input->W;
    uint32_t pH = args->weights->H;
    uint32_t pW = args->weights->W;
    uint32_t H_out = args->output->H;
    uint32_t W_out = args->output->W;

    uint32_t blockSize = (C_in + NUM_CORES - 1) / NUM_CORES;
    uint32_t start = pi_core_id() * blockSize;
    uint32_t stop = start + blockSize > C_in ? C_in : start + blockSize;

    for (int ch = start; ch < stop; ch++) {
        for (int hk = 0; hk < pH; hk++) {
            for (int wk = 0; wk < pW; wk++) {
                int idx = wk + hk * pW + ch * pH * pW;
                float old_val = coeffDiff[idx];
                float temp = 0;

                for (int ho = 0; ho < H_out; ho++) {
                    for (int wo = 0; wo < W_out; wo++) {

                        temp += inData[wk + wo + (hk + ho) * W_in + ch * H_in * W_in] *
                                outDiff[wo + ho * W_out + ch * H_out * W_out];
                    }
                }

                coeffDiff[idx] += temp;
            }
        }
    }
}


// Weight grad kernel for DepthWise Convolution with padding and arbitrary stride
void dw_kernel_weight_grad_padded(void *kernel_DW_args) {
    struct kernel_DW_args *args = (struct kernel_DW_args *) kernel_DW_args;

    float *inData = args->input->data;
    float *coeffDiff = args->weights->diff;
    float *outDiff = args->output->diff;

    int C_in  = (int) args->input->C;
    int H_in  = (int) args->input->H;
    int W_in  = (int) args->input->W;
    int pH    = (int) args->weights->H;
    int pW    = (int) args->weights->W;
    int H_out = (int) args->output->H;
    int W_out = (int) args->output->W;

    int Upad     = args->Upad;
    int Lpad     = args->Lpad;
    int stride_h = args->stride_h;
    int stride_w = args->stride_w;

    int blockSize = (C_in + NUM_CORES - 1) / NUM_CORES;
    int start = pi_core_id() * blockSize;
    int stop  = start + blockSize > C_in ? C_in : start + blockSize;

    /* Precompute valid output-row range for each kernel row (ho_min/ho_max per hk). */
    for (int ch = start; ch < stop; ch++) {
        int ch_in_off  = ch * H_in  * W_in;
        int ch_out_off = ch * H_out * W_out;
        for (int hk = 0; hk < pH; hk++) {
            /* h_in = ho*stride_h + hk - Upad  must be in [0, H_in) */
            int ho_min = (Upad - hk + stride_h - 1) / stride_h;
            if (ho_min < 0) ho_min = 0;
            int ho_max = (H_in - 1 + Upad - hk) / stride_h + 1;
            if (ho_max > H_out) ho_max = H_out;

            for (int wk = 0; wk < pW; wk++) {
                /* w_in = wo*stride_w + wk - Lpad  must be in [0, W_in) */
                int wo_min = (Lpad - wk + stride_w - 1) / stride_w;
                if (wo_min < 0) wo_min = 0;
                int wo_max = (W_in - 1 + Lpad - wk) / stride_w + 1;
                if (wo_max > W_out) wo_max = W_out;

                int idx = wk + hk * pW + ch * pH * pW;
                float temp = 0;

                for (int ho = ho_min; ho < ho_max; ho++) {
                    int h_in = ho * stride_h + hk - Upad;
                    for (int wo = wo_min; wo < wo_max; wo++) {
                        int w_in = wo * stride_w + wk - Lpad;
                        temp += inData[w_in + h_in * W_in + ch_in_off] *
                                outDiff[wo + ho * W_out + ch_out_off];
                    }
                }

                coeffDiff[idx] += temp;
            }
        }
    }
}


// Naive input grad kernel for DepthWise Convolution
void dw_kernel_input_grad(void *kernel_DW_args) {
    struct kernel_DW_args *args = (struct kernel_DW_args *) kernel_DW_args;

    float *inDiff = args->input->diff;
    float *coeffData = args->weights->data;
    float *outDiff = args->output->diff;

    uint32_t C_in = args->input->C;
    uint32_t H_in = args->input->H;
    uint32_t W_in = args->input->W;
    uint32_t pH = args->weights->H;
    uint32_t pW = args->weights->W;
    uint32_t H_out = args->output->H;
    uint32_t W_out = args->output->W;

    uint32_t blockSize = (C_in + NUM_CORES - 1) / NUM_CORES;
    uint32_t start = pi_core_id() * blockSize;
    uint32_t stop = start + blockSize > C_in ? C_in : start + blockSize;

    for (int ch = 0; ch < C_in; ch++) {
        for (int hin = 0; hin < H_in; hin++) {
            int ho = hin - pH + 1;

            for (int win = 0; win < W_in; win++) {
                int wo = win - pW + 1;
                float temp = 0;

                for (int hk = 0; hk < pH; hk++) {
                    for (int wk = 0; wk < pW; wk++) {
                        if ((wo + wk >= 0) && (ho + hk >= 0) && (wo + wk < W_out) && (ho + hk < H_out)) {

                            temp += coeffData[(pW - 1 - wk) + (pH - 1 - hk) * pW + ch * pH * pW] *
                                    outDiff[(wo + wk) + (ho + hk) * W_out + ch * H_out * W_out];
                        }
                    }
                }

                inDiff[win + hin * W_in + ch * H_in * W_in] = temp;
            }
        }
    }
}


// Input grad kernel for DepthWise Convolution with padding and arbitrary stride
void dw_kernel_input_grad_padded(void *kernel_DW_args) {
    struct kernel_DW_args *args = (struct kernel_DW_args *) kernel_DW_args;

    float *inDiff    = args->input->diff;
    float *coeffData = args->weights->data;
    float *outDiff   = args->output->diff;

    int C_in  = (int) args->input->C;
    int H_in  = (int) args->input->H;
    int W_in  = (int) args->input->W;
    int pH    = (int) args->weights->H;
    int pW    = (int) args->weights->W;
    int H_out = (int) args->output->H;
    int W_out = (int) args->output->W;
    int Upad     = args->Upad;
    int Lpad     = args->Lpad;
    int stride_h = args->stride_h;
    int stride_w = args->stride_w;

    int blockSize = (C_in + NUM_CORES - 1) / NUM_CORES;
    int start = pi_core_id() * blockSize;
    int stop  = start + blockSize > C_in ? C_in : start + blockSize;

    /* For each input position (ch, hin, win):
     *   dX[ch,hin,win] = sum_{valid ho,wo} dY[ch,ho,wo] * W[ch, hin+Upad-ho*sh, win+Lpad-wo*sw]
     *
     * Precompute valid ho/wo ranges to avoid branch-heavy innermost loops
     * (branches in innermost loops miscompile under GCC -O3 -ffast-math on RISC-V).
     */
    for (int ch = start; ch < stop; ch++) {
        int ch_in_off  = ch * H_in  * W_in;
        int ch_out_off = ch * H_out * W_out;
        int ch_w_off   = ch * pH * pW;

        for (int hin = 0; hin < H_in; hin++) {
            /* ho range: hk = hin+Upad - ho*sh must be in [0, pH) */
            int a_h   = hin + Upad - pH + 1;
            int ho_min = (a_h <= 0) ? 0 : (a_h + stride_h - 1) / stride_h;
            int ho_max = (hin + Upad) / stride_h + 1;
            if (ho_max > H_out) ho_max = H_out;

            for (int win = 0; win < W_in; win++) {
                /* wo range: wk = win+Lpad - wo*sw must be in [0, pW) */
                int a_w   = win + Lpad - pW + 1;
                int wo_min = (a_w <= 0) ? 0 : (a_w + stride_w - 1) / stride_w;
                int wo_max = (win + Lpad) / stride_w + 1;
                if (wo_max > W_out) wo_max = W_out;

                float temp = 0;
                for (int ho = ho_min; ho < ho_max; ho++) {
                    int hk      = hin + Upad - ho * stride_h;
                    int out_row = ho * W_out + ch_out_off;
                    int w_row   = hk * pW    + ch_w_off;
                    for (int wo = wo_min; wo < wo_max; wo++) {
                        int wk = win + Lpad - wo * stride_w;
                        temp += coeffData[wk + w_row] * outDiff[wo + out_row];
                    }
                }
                inDiff[win + hin * W_in + ch_in_off] = temp;
            }
        }
    }
}


// Tile-aware input grad kernel for DepthWise Convolution.
// Same gather pattern as dw_kernel_input_grad_padded but with tile offsets:
// H_in/W_in/H_out/W_out are tile-local dimensions; offset_in/out_h/w give the
// global position of the tile so that padding and stride calculations remain
// correct across spatial tiles.
void dw_kernel_input_grad_padded_tiled(void *kernel_DW_args) {
    struct kernel_DW_args *args = (struct kernel_DW_args *) kernel_DW_args;

    float *inDiff    = args->input->diff;
    float *coeffData = args->weights->data;
    float *outDiff   = args->output->diff;

    int C_in  = (int) args->input->C;
    int H_in  = (int) args->input->H;   // tile height of dX
    int W_in  = (int) args->input->W;   // tile width  of dX
    int pH    = (int) args->weights->H;
    int pW    = (int) args->weights->W;
    int H_out = (int) args->output->H;  // tile height of dY
    int W_out = (int) args->output->W;  // tile width  of dY
    int Upad     = args->Upad;
    int Lpad     = args->Lpad;
    int stride_h = args->stride_h;
    int stride_w = args->stride_w;

    int off_in_h  = args->offset_in_h;
    int off_in_w  = args->offset_in_w;
    int off_out_h = args->offset_out_h;
    int off_out_w = args->offset_out_w;

    int blockSize = (C_in + NUM_CORES - 1) / NUM_CORES;
    int start = pi_core_id() * blockSize;
    int stop  = start + blockSize > C_in ? C_in : start + blockSize;

    /* For each dX position (ch, hin_local, win_local) in the tile we compute
     * the global coordinate and derive which dY positions (ho, wo) contribute.
     * ho/wo are then mapped back to tile-local dY indices for the array access.
     *
     * dX[ch,hin,win] = sum_{valid ho,wo} dY[ch,ho,wo] * W_rot[ch,hk,wk]
     *   where hk = hin_global + Upad - ho_global * stride_h
     */
    for (int ch = start; ch < stop; ch++) {
        int ch_in_off  = ch * H_in  * W_in;
        int ch_out_off = ch * H_out * W_out;
        int ch_w_off   = ch * pH * pW;

        for (int hin = 0; hin < H_in; hin++) {
            int hin_g = hin + off_in_h;  // global coordinate

            /* ho_global range: hk = hin_g + Upad - ho_g * sh must be in [0, pH) */
            int a_h    = hin_g + Upad - pH + 1;
            int ho_g_min = (a_h <= 0) ? 0 : (a_h + stride_h - 1) / stride_h;
            int ho_g_max = (hin_g + Upad) / stride_h + 1;

            /* Convert global ho range to tile-local dY indices */
            int ho_min = ho_g_min - off_out_h;
            int ho_max = ho_g_max - off_out_h;
            if (ho_min < 0) ho_min = 0;
            if (ho_max > H_out) ho_max = H_out;

            for (int win = 0; win < W_in; win++) {
                int win_g = win + off_in_w;  // global coordinate

                /* wo_global range */
                int a_w    = win_g + Lpad - pW + 1;
                int wo_g_min = (a_w <= 0) ? 0 : (a_w + stride_w - 1) / stride_w;
                int wo_g_max = (win_g + Lpad) / stride_w + 1;

                int wo_min = wo_g_min - off_out_w;
                int wo_max = wo_g_max - off_out_w;
                if (wo_min < 0) wo_min = 0;
                if (wo_max > W_out) wo_max = W_out;

                float temp = 0;
                for (int ho = ho_min; ho < ho_max; ho++) {
                    int ho_g   = ho + off_out_h;
                    int hk     = hin_g + Upad - ho_g * stride_h;
                    int out_row = ho * W_out + ch_out_off;
                    int w_row   = hk * pW    + ch_w_off;
                    for (int wo = wo_min; wo < wo_max; wo++) {
                        int wo_g = wo + off_out_w;
                        int wk   = win_g + Lpad - wo_g * stride_w;
                        temp += coeffData[wk + w_row] * outDiff[wo + out_row];
                    }
                }
                inDiff[win + hin * W_in + ch_in_off] = temp;
            }
        }
    }
}


/** CONV2D KERNELS **/
void naive_conv2d_fw_kernel_CHW(void *matMul_args) {
    struct matMul_args *args = (struct matMul_args *) matMul_args;

    float *__restrict__ inData = args->A;
    float *__restrict__ coeffData = args->B;
    float *__restrict__ biasData = args->bias;
    float *__restrict__ outData = args->C;


    const uint32_t H_in = args->H;
    const uint32_t W_in = args->W;
    const uint32_t pW = args->pW;
    const uint32_t pH = args->pH;
    const uint32_t C_in = args->pCin;
    const uint32_t C_out = args->pCout;

    uint32_t h_str = args->stride_h;
    uint32_t w_str = args->stride_w;

    uint32_t Lpad = args->Lpad;
    uint32_t Rpad = args->Rpad;
    uint32_t Upad = args->Upad;
    uint32_t Dpad = args->Dpad;

    const uint32_t H_out = (H_in - pH + Upad + Dpad) / h_str + 1;
    const uint32_t W_out = (W_in - pW + Lpad + Rpad) / w_str + 1;

    const uint32_t USE_BIASES = args->USE_BIASES;

    const uint32_t blockSize = (C_out + NUM_CORES - 1) / NUM_CORES;
    const uint32_t start = pi_core_id() * blockSize;
    const uint32_t stop = start + blockSize > C_out ? C_out : start + blockSize;

    int padding = Lpad + Rpad + Upad + Dpad;

    if (padding == 0) {
        for (uint32_t co = start; co < stop; co++) {
            for (uint32_t ho = 0; ho < H_out; ho++) {
                for (uint32_t wo = 0; wo < W_out; wo++) {
                    float temp = 0;
                    // Receptive field
                    for (uint32_t ci = 0; ci < C_in; ci++) {
                        for (uint32_t hk = 0; hk < pH; hk++) {
                            for (uint32_t wk = 0; wk < pW; wk++) {
                                // Indices
                                int in_idx = w_str * wo + wk + (h_str * ho + hk) * W_in + ci * H_in * W_in;
                                int coeff_idx = wk + hk * pW + ci * pH * pW + co * C_in * pH * pW;

                                temp += inData[in_idx] * coeffData[coeff_idx];
                            }
                        }
                    }
                    outData[wo + ho * W_out + co * H_out * W_out] = temp;

                    // Handle biases
                    if (USE_BIASES == 1) {
                        outData[wo + ho * W_out + co * H_out * W_out] += biasData[co];
                    }
                }
            }
        }
    } else {
        for (uint32_t co = start; co < stop; co++) {
            for (uint32_t ho = 0; ho < H_out; ho++) {
                for (uint32_t wo = 0; wo < W_out; wo++) {
                    float temp = 0;

                    // Receptive field
                    for (uint32_t ci = 0; ci < C_in; ci++) {
                        for (uint32_t hk = 0; hk < pH; hk++) {
                            for (uint32_t wk = 0; wk < pW; wk++) {
                                // Pad conditions
                                int pad_cond_h = h_str * ho + hk - Upad;
                                int pad_cond_w = w_str * wo + wk - Lpad;

                                if ((pad_cond_h >= 0) && (pad_cond_w >= 0) && (pad_cond_h < H_in) &&
                                    (pad_cond_w < W_in)) {
                                    int in_idx = (w_str * wo + wk - Lpad) + (h_str * ho + hk - Upad) * W_in +
                                                 ci * H_in * W_in;
                                    int ker_idx = wk + hk * pW + ci * pH * pW + co * C_in * pH * pW;

                                    temp += inData[in_idx] * coeffData[ker_idx];
                                }
                            }
                        }
                    }
                    outData[wo + ho * W_out + co * H_out * W_out] = temp;

                    // Handle biases
                    if (USE_BIASES == 1) {
                        outData[wo + ho * W_out + co * H_out * W_out] += biasData[co];
                    }
                }
            }
        }
    }

    if (USE_BIASES != 0 && USE_BIASES != 1) {
        printf("[naive_conv2d_fw_kernel_CHW:] Invalid selection of the bias option (1 or 0 - use biases or not). Actual value: %d. Biases not used, even if provided!\n",
               USE_BIASES);
    }
}


void naive_conv2d_param_grad_kernel_CHW(void *matMul_args) {
    struct matMul_args *args = (struct matMul_args *) matMul_args;

    float *__restrict__ inData = args->A;
    float *__restrict__ coeffDiff = args->B;
    float *__restrict__ biasDiff = args->bias;
    float *__restrict__ outDiff = args->C;


    const uint32_t H_in = args->H;
    const uint32_t W_in = args->W;
    const uint32_t C_in = args->pCin;

    const uint32_t pW = args->pW;
    const uint32_t pH = args->pH;

    uint32_t h_str = args->stride_h;
    uint32_t w_str = args->stride_w;

    uint32_t Lpad = args->Lpad;
    uint32_t Rpad = args->Rpad;
    uint32_t Upad = args->Upad;
    uint32_t Dpad = args->Dpad;

    const uint32_t USE_BIASES = args->USE_BIASES;

    const uint32_t H_out = (H_in - pH + Upad + Dpad) / h_str + 1;
    const uint32_t W_out = (W_in - pW + Lpad + Rpad) / w_str + 1;
    const uint32_t C_out = args->pCout;

    const uint32_t blockSize = (C_out + NUM_CORES - 1) / NUM_CORES;
    const uint32_t start = pi_core_id() * blockSize;
    const uint32_t stop = start + blockSize > C_out ? C_out : start + blockSize;

    int padding = Lpad + Rpad + Upad + Dpad;

    if (padding == 0) {
        for (uint32_t co = start; co < stop; co++) {
            for (uint32_t hk = 0; hk < pH; hk++) {
                for (uint32_t wk = 0; wk < pW; wk++) {
                    for (uint32_t ci = 0; ci < C_in; ci++) {
                        float temp = 0;
                        float bias_temp = 0;

                        for (uint32_t ho = 0; ho < H_out; ho++) {
                            for (uint32_t wo = 0; wo < W_out; wo++) {
                                // Indices
                                int out_idx = wo + ho * W_out + co * H_out * W_out;
                                int in_idx = w_str * wo + wk + (h_str * ho + hk) * W_in + ci * H_in * W_in;

                                temp += outDiff[out_idx] * inData[in_idx];

                                if (USE_BIASES == 1) bias_temp += outDiff[out_idx];
                            }
                        }
                        coeffDiff[wk + hk * pW + ci * pH * pW + co * pH * pW * C_in] = temp;

                        if (USE_BIASES == 1) {
                            biasDiff[co] = bias_temp;
                        }
                    }
                }
            }
        }
    } else {
        for (uint32_t co = start; co < stop; co++) {
            for (uint32_t hk = 0; hk < pH; hk++) {
                for (uint32_t wk = 0; wk < pW; wk++) {
                    for (uint32_t ci = 0; ci < C_in; ci++) {
                        float temp = 0;
                        float bias_temp = 0;

                        for (uint32_t ho = 0; ho < H_out; ho++) {
                            for (uint32_t wo = 0; wo < W_out; wo++) {
                                // Pad conditions
                                int pad_cond_h = h_str * ho + hk - Upad;
                                int pad_cond_w = w_str * wo + wk - Lpad;

                                if ((pad_cond_h >= 0) && (pad_cond_w >= 0) && (pad_cond_h < H_in) &&
                                    (pad_cond_w < W_in)) {
                                    int out_idx = wo + ho * W_out + co * H_out * W_out;
                                    int in_idx = (w_str * wo + wk - Lpad) + (h_str * ho + hk - Upad) * W_in +
                                                 ci * H_in * W_in;

                                    temp += outDiff[out_idx] * inData[in_idx];

                                    if (USE_BIASES == 1) bias_temp += outDiff[out_idx];
                                }
                            }
                        }
                        coeffDiff[wk + hk * pW + ci * pH * pW + co * pH * pW * C_in] = temp;

                        if (USE_BIASES == 1) {
                            biasDiff[co] = bias_temp;
                        }
                    }
                }
            }
        }
    }

    if (USE_BIASES != 0 && USE_BIASES != 1) {
        printf("[naive_conv2d_param_grad_kernel_CHW:] Invalid selection of the bias option (1 or 0 - use biases or not). Actual value: %d. Biases not used, even if provided!\n",
               USE_BIASES);
    }
}


void naive_conv2d_in_grad_kernel_CHW(void *matMul_args) {
    struct matMul_args *args = (struct matMul_args *) matMul_args;

    float *__restrict__ inDiff = args->A;
    float *__restrict__ coeffData = args->B;
    float *__restrict__ outDiff = args->C;

    const uint32_t USE_BIASES = args->USE_BIASES;

    const uint32_t H_in = args->H;
    const uint32_t W_in = args->W;
    const uint32_t pW = args->pW;
    const uint32_t pH = args->pH;
    const uint32_t C_in = args->pCin;
    const uint32_t C_out = args->pCout;

    uint32_t h_str = args->stride_h;
    uint32_t w_str = args->stride_w;
    uint32_t Lpad = args->Lpad;
    uint32_t Rpad = args->Rpad;
    uint32_t Upad = args->Upad;
    uint32_t Dpad = args->Dpad;

    const uint32_t H_out = (H_in - pH + Upad + Dpad) / h_str + 1;
    const uint32_t W_out = (W_in - pW + Lpad + Rpad) / w_str + 1;
    const uint32_t pHW = pH * pW - 1;

    const uint32_t blockSize = (C_in + NUM_CORES - 1) / NUM_CORES;
    const uint32_t start = pi_core_id() * blockSize;
    const uint32_t stop = start + blockSize > C_in ? C_in : start + blockSize;

    int padding = Lpad + Rpad + Upad + Dpad;
    int stride = h_str + w_str;

    if ((padding == 0) && (stride <= 2)) {

        for (uint32_t ci = start; ci < stop; ci++) {
            for (uint32_t hi = 0; hi < H_in; hi++) {
                for (uint32_t wi = 0; wi < W_in; wi++) {
                    float temp = 0;
                    for (uint32_t co = 0; co < C_out; co++) {
                        for (uint32_t hk = 0; hk < pH; hk++) {
                            for (uint32_t wk = 0; wk < pW; wk++) {
                                // Padding conditions
                                int h_padded = hi + hk - (pH - 1);
                                int w_padded = wi + wk - (pW - 1);
                                // Indices
                                int ker_idx = (pHW - wk - hk * pW) + ci * pW * pH + co * pW * pH * C_in;
                                int out_idx = w_padded + (h_padded) * W_out + co * H_out * W_out;

                                if ((h_padded >= 0) && (w_padded >= 0) && (h_padded <= H_out - (pH - 2)) &&
                                    (w_padded <= W_out - (pW - 2))) {
                                    temp += coeffData[ker_idx] * outDiff[out_idx];
                                }
                            }
                        }
                    }
                    inDiff[wi + hi * W_in + ci * H_in * W_in] = temp;
                }
            }
        }

    } else {
        int h_start = pH - 1 - Lpad;
        int w_start = pW - 1 - Upad;

        for (uint32_t ci = start; ci < stop; ci++) {
            for (uint32_t hi = 0; hi < H_in; hi++) {
                for (uint32_t wi = 0; wi < W_in; wi++) {
                    float temp = 0;

                    for (uint32_t co = 0; co < C_out; co++) {
                        for (uint32_t hk = 0; hk < pH; hk++) {
                            for (uint32_t wk = 0; wk < pW; wk++) {
                                // Indices
                                int ker_idx = (pHW - wk - hk * pW) + ci * pW * pH + co * pW * pH * C_in;

                                // Border conditions
                                int bord_h = (hi + hk - h_start) / h_str;
                                int bord_w = (wi + wk - w_start) / w_str;
                                int borders = (bord_h < H_out) && (bord_w < W_out);

                                float o_grad = 0;
                                float k_dat = coeffData[ker_idx];

                                if (((hi + hk - h_start) % h_str == 0) && ((wi + wk - w_start) % w_str == 0) &&
                                    borders == 1) {
                                    int out_idx = bord_w + (bord_h) * W_out + co * H_out * W_out;

                                    o_grad = outDiff[out_idx];
                                    temp += k_dat * o_grad;
                                }
                            }
                        }
                    }

                    inDiff[wi + hi * W_in + ci * H_in * W_in] = temp;
                }
            }
        }
    }

    if (USE_BIASES != 0 && USE_BIASES != 1) {
        printf("[naive_conv2d_param_grad_kernel_CHW:] Invalid selection of the bias option (1 or 0 - use biases or not). Actual value: %d. Current step not affected by this.\n",
               USE_BIASES);
    }
}




// Tile-aware naive conv2d input gradient kernel (CHW format).
// Same gather pattern as naive_conv2d_in_grad_kernel_CHW but with tile offsets:
// H/W are tile-local dimensions; offset_in/out_h/w give the global position.
void naive_conv2d_in_grad_kernel_CHW_tiled(void *matMul_args) {
    struct matMul_args *args = (struct matMul_args *) matMul_args;

    float *__restrict__ inDiff = args->A;     // dX tile
    float *__restrict__ coeffData = args->B;  // W (full)
    float *__restrict__ outDiff = args->C;    // dY tile

    const int H_in = args->H;     // tile height of dX
    const int W_in = args->W;     // tile width of dX
    const int pW = args->pW;
    const int pH = args->pH;
    const int C_in = args->pCin;
    const int C_out = args->pCout;

    const int stride_h = args->stride_h;
    const int stride_w = args->stride_w;
    const int Upad = args->Upad;
    const int Lpad = args->Lpad;

    const int off_in_h  = args->offset_in_h;
    const int off_in_w  = args->offset_in_w;
    const int off_out_h = args->offset_out_h;
    const int off_out_w = args->offset_out_w;

    // Derive dY tile dimensions from blob (passed via args->C which points
    // to outDiff; H_out/W_out are not in matMul_args, so we compute them
    // from the Conv2D_args that set up this call — the wrapper passes them
    // through the H_out/W_out fields repurposed into N/K).
    const int H_out = args->N;  // repurposed: wrapper stores H_out_tile here
    const int W_out = args->K;  // repurposed: wrapper stores W_out_tile here

    const int blockSize = (C_in + NUM_CORES - 1) / NUM_CORES;
    const int start = pi_core_id() * blockSize;
    int stop  = start + blockSize;
    if (stop > C_in) stop = C_in;

    for (int ci = start; ci < stop; ci++) {
        int ch_in_off  = ci * H_in * W_in;
        int ch_w_base  = ci * pH * pW;  // W layout: [Cout, Cin, pH, pW]

        for (int hi = 0; hi < H_in; hi++) {
            int hi_g = hi + off_in_h;  // global coordinate

            for (int wi = 0; wi < W_in; wi++) {
                int wi_g = wi + off_in_w;

                float temp = 0;
                for (int co = 0; co < C_out; co++) {
                    int ch_out_off = co * H_out * W_out;
                    // W index base for this (co, ci) pair
                    // W layout: [Cout][Cin][pH][pW], kernel is flipped for
                    // transposed conv
                    int w_co_ci = co * C_in * pH * pW + ch_w_base;

                    for (int hk = 0; hk < pH; hk++) {
                        // ho_global * stride = hi_global + Upad - hk
                        int num_h = hi_g + Upad - hk;
                        if (num_h < 0 || num_h % stride_h != 0)
                            continue;
                        int ho_g = num_h / stride_h;
                        int ho = ho_g - off_out_h;  // tile-local
                        if (ho < 0 || ho >= H_out)
                            continue;

                        for (int wk = 0; wk < pW; wk++) {
                            int num_w = wi_g + Lpad - wk;
                            if (num_w < 0 || num_w % stride_w != 0)
                                continue;
                            int wo_g = num_w / stride_w;
                            int wo = wo_g - off_out_w;
                            if (wo < 0 || wo >= W_out)
                                continue;

                            // Flipped kernel: W[co, ci, pH-1-hk, pW-1-wk]
                            int ker_idx = w_co_ci + (pH - 1 - hk) * pW + (pW - 1 - wk);
                            int out_idx = ch_out_off + ho * W_out + wo;

                            temp += coeffData[ker_idx] * outDiff[out_idx];
                        }
                    }
                }
                inDiff[ch_in_off + hi * W_in + wi] = temp;
            }
        }
    }
}
/** CONV2D OPTIMIZED VERSIONS **/
void naive_conv2d_fw_kernel_CHW_k3x3_s2_p1(void *matMul_args) {
    struct matMul_args *args = (struct matMul_args *) matMul_args;

    float *__restrict__ inData = args->A;
    float *__restrict__ coeffData = args->B;
    float *__restrict__ biasData = args->bias;
    float *__restrict__ outData = args->C;


    const uint32_t H_in = args->H;
    const uint32_t W_in = args->W;
    const uint32_t C_in = args->pCin;

    const uint32_t H_out = (H_in - 1) / 2 + 1;
    const uint32_t W_out = (W_in - 1) / 2 + 1;
    const uint32_t C_out = args->pCout;

    const uint32_t USE_BIASES = args->USE_BIASES;

    const uint32_t blockSize = (C_out + NUM_CORES - 1) / NUM_CORES;
    const uint32_t start = pi_core_id() * blockSize;
    const uint32_t stop = start + blockSize > C_out ? C_out : start + blockSize;

    for (uint32_t co = start; co < stop; co++) {
        for (uint32_t ho = 0; ho < H_out; ho++) {
            for (uint32_t wo = 0; wo < W_out; wo++) {
                float temp = 0;

                // Receptive field
                for (uint32_t ci = 0; ci < C_in; ci++) {
                    for (uint32_t hk = 0; hk < 3; hk++) {
                        for (uint32_t wk = 0; wk < 3; wk++) {
                            // Pad conditions
                            int pad_cond_h = 2 * ho + hk - 1;
                            int pad_cond_w = 2 * wo + wk - 1;

                            if ((pad_cond_h >= 0) && (pad_cond_w >= 0) && (pad_cond_h < H_in) && (pad_cond_w < W_in)) {
                                int in_idx = (2 * wo + wk - 1) + (2 * ho + hk - 1) * W_in + ci * H_in * W_in;
                                int ker_idx = wk + hk * 3 + ci * 9 + co * C_in * 9;

                                temp += inData[in_idx] * coeffData[ker_idx];
                            }
                        }
                    }
                }
                outData[wo + ho * W_out + co * H_out * W_out] = temp;

                if (USE_BIASES == 1) {
                    outData[wo + ho * W_out + co * H_out * W_out] += biasData[co];
                }
            }
        }
    }

    if (USE_BIASES != 0 && USE_BIASES != 1) {
        printf("[naive_conv2d_fw_kernel_CHW_k3x3_s2_p1:] Invalid selection of the bias option (1 or 0 - use biases or not). Actual value: %d. Biases not used, even if provided!\n",
               USE_BIASES);
    }
}


void naive_conv2d_param_grad_kernel_CHW_k3x3_s2_p1(void *matMul_args) {
    struct matMul_args *args = (struct matMul_args *) matMul_args;

    float *__restrict__ inData = args->A;
    float *__restrict__ coeffDiff = args->B;
    float *__restrict__ biasDiff = args->bias;
    float *__restrict__ outDiff = args->C;

    const uint32_t H_in = args->H;
    const uint32_t W_in = args->W;
    const uint32_t C_in = args->pCin;
    const uint32_t C_out = args->pCout;

    const uint32_t H_out = (H_in - 1) / 2 + 1;
    const uint32_t W_out = (W_in - 1) / 2 + 1;

    const uint32_t USE_BIASES = args->USE_BIASES;

    const uint32_t blockSize = (C_out + NUM_CORES - 1) / NUM_CORES;
    const uint32_t start = pi_core_id() * blockSize;
    const uint32_t stop = start + blockSize > C_out ? C_out : start + blockSize;

    for (uint32_t co = start; co < stop; co++) {
        for (uint32_t hk = 0; hk < 3; hk++) {
            for (uint32_t wk = 0; wk < 3; wk++) {
                for (uint32_t ci = 0; ci < C_in; ci++) {
                    float temp = 0;
                    float bias_temp = 0;

                    for (uint32_t ho = 0; ho < H_out; ho++) {
                        for (uint32_t wo = 0; wo < W_out; wo++) {
                            // Pad conditions
                            int pad_cond_h = 2 * ho + hk - 1;
                            int pad_cond_w = 2 * wo + wk - 1;

                            if ((pad_cond_h >= 0) && (pad_cond_w >= 0) && (pad_cond_h < H_in) && (pad_cond_w < W_in)) {
                                int out_idx = wo + ho * W_out + co * H_out * W_out;
                                int in_idx = (2 * wo + wk - 1) + (2 * ho + hk - 1) * W_in + ci * H_in * W_in;

                                temp += outDiff[out_idx] * inData[in_idx];

                                if (USE_BIASES == 1) bias_temp += outDiff[out_idx];
                            }
                        }
                    }
                    coeffDiff[wk + hk * 3 + ci * 9 + co * 9 * C_in] = temp;

                    if (USE_BIASES == 1) {
                        biasDiff[co] = bias_temp;
                    }
                }
            }
        }
    }

    if (USE_BIASES != 0 && USE_BIASES != 1) {
        printf("[naive_conv2d_param_grad_kernel_CHW_k3x3_s2_p1:] Invalid selection of the bias option (1 or 0 - use biases or not). Actual value: %d. Biases not used, even if provided!\n",
               USE_BIASES);
    }
}


void naive_conv2d_in_grad_kernel_CHW_k3x3_s2_p1(void *matMul_args) {
    struct matMul_args *args = (struct matMul_args *) matMul_args;

    float *__restrict__ inDiff = args->A;
    float *__restrict__ coeffData = args->B;
    float *__restrict__ outDiff = args->C;

    const uint32_t USE_BIASES = args->USE_BIASES;

    const uint32_t H_in = args->H;
    const uint32_t W_in = args->W;
    const uint32_t C_in = args->pCin;
    const uint32_t C_out = args->pCout;

    const uint32_t H_out = (H_in - 1) / 2 + 1;
    const uint32_t W_out = (W_in - 1) / 2 + 1;
    const uint32_t pHW = 8;

    const uint32_t blockSize = (C_in + NUM_CORES - 1) / NUM_CORES;
    const uint32_t start = pi_core_id() * blockSize;
    const uint32_t stop = start + blockSize > C_in ? C_in : start + blockSize;

    for (uint32_t ci = start; ci < stop; ci++) {
        for (uint32_t hi = 0; hi < H_in; hi++) {
            for (uint32_t wi = 0; wi < W_in; wi++) {
                float temp = 0;
                for (uint32_t co = 0; co < C_out; co++) {
                    for (uint32_t hk = 0; hk < 3; hk++) {
                        for (uint32_t wk = 0; wk < 3; wk++) {
                            // Border conditions
                            int bord_h = (hi + hk - 1) / 2;
                            int bord_w = (wi + wk - 1) / 2;
                            int borders = (bord_h < H_out) && (bord_w < W_out);

                            if (((hi + hk - 1) % 2 == 0) && ((wi + wk - 1) % 2 == 0) && borders == 1) {
                                int out_idx = bord_w + (bord_h) * W_out + co * H_out * W_out;
                                int ker_idx = (8 - wk - hk * 3) + ci * 9 + co * 9 * C_in;

                                float k_dat = coeffData[ker_idx];
                                float o_grad = outDiff[out_idx];

                                temp += k_dat * o_grad;
                            }
                        }
                    }
                }

                inDiff[wi + hi * W_in + ci * H_in * W_in] = temp;
            }
        }
    }

    if (USE_BIASES != 0 && USE_BIASES != 1) {
        printf("[naive_conv2d_in_grad_kernel_CHW_k3x3_s2_p1:] Invalid selection of the bias option (1 or 0 - use biases or not). Actual value: %d. Current step not affected by this.\n",
               USE_BIASES);
    }
}


void naive_conv2d_fw_kernel_CHW_k5x5_s2_p1(void *matMul_args) {
    struct matMul_args *args = (struct matMul_args *) matMul_args;

    float *__restrict__ inData = args->A;
    float *__restrict__ coeffData = args->B;
    float *__restrict__ biasData = args->bias;
    float *__restrict__ outData = args->C;

    const uint32_t H_in = args->H;
    const uint32_t W_in = args->W;
    const uint32_t C_in = args->pCin;

    const uint32_t H_out = (H_in - 3) / 2 + 1;
    const uint32_t W_out = (W_in - 3) / 2 + 1;
    const uint32_t C_out = args->pCout;

    const uint32_t USE_BIASES = args->USE_BIASES;

    const uint32_t blockSize = (C_out + NUM_CORES - 1) / NUM_CORES;
    const uint32_t start = pi_core_id() * blockSize;
    const uint32_t stop = start + blockSize > C_out ? C_out : start + blockSize;

    for (uint32_t co = start; co < stop; co++) {
        for (uint32_t ho = 0; ho < H_out; ho++) {
            for (uint32_t wo = 0; wo < W_out; wo++) {
                float temp = 0;

                // Receptive field
                for (uint32_t ci = 0; ci < C_in; ci++) {
                    for (uint32_t hk = 0; hk < 5; hk++) {
                        for (uint32_t wk = 0; wk < 5; wk++) {
                            // Pad conditions
                            int pad_cond_h = 2 * ho + hk - 1;
                            int pad_cond_w = 2 * wo + wk - 1;

                            if ((pad_cond_h >= 0) && (pad_cond_w >= 0) && (pad_cond_h < H_in) && (pad_cond_w < W_in)) {
                                int in_idx = (2 * wo + wk - 1) + (2 * ho + hk - 1) * W_in + ci * H_in * W_in;
                                int ker_idx = wk + hk * 5 + ci * 25 + co * C_in * 25;

                                temp += inData[in_idx] * coeffData[ker_idx];
                            }
                        }
                    }
                }
                outData[wo + ho * W_out + co * H_out * W_out] = temp;

                if (USE_BIASES == 1) {
                    outData[wo + ho * W_out + co * H_out * W_out] += biasData[co];
                }
            }
        }
    }

    if (USE_BIASES != 0 && USE_BIASES != 1) {
        printf("[naive_conv2d_fw_kernel_CHW_k5x5_s2_p1:] Invalid selection of the bias option (1 or 0 - use biases or not). Actual value: %d. Biases not used, even if provided!\n",
               USE_BIASES);
    }
}


void naive_conv2d_param_grad_kernel_CHW_k5x5_s2_p1(void *matMul_args) {
    struct matMul_args *args = (struct matMul_args *) matMul_args;

    float *__restrict__ inData = args->A;
    float *__restrict__ coeffDiff = args->B;
    float *__restrict__ biasDiff = args->bias;
    float *__restrict__ outDiff = args->C;


    const uint32_t H_in = args->H;
    const uint32_t W_in = args->W;
    const uint32_t C_in = args->pCin;

    const uint32_t H_out = (H_in - 3) / 2 + 1;
    const uint32_t W_out = (W_in - 3) / 2 + 1;
    const uint32_t C_out = args->pCout;

    const uint32_t USE_BIASES = args->USE_BIASES;

    const uint32_t blockSize = (C_out + NUM_CORES - 1) / NUM_CORES;
    const uint32_t start = pi_core_id() * blockSize;
    const uint32_t stop = start + blockSize > C_out ? C_out : start + blockSize;

    for (uint32_t co = start; co < stop; co++) {
        for (uint32_t hk = 0; hk < 5; hk++) {
            for (uint32_t wk = 0; wk < 5; wk++) {
                for (uint32_t ci = 0; ci < C_in; ci++) {
                    float temp = 0;
                    float bias_temp = 0;

                    for (uint32_t ho = 0; ho < H_out; ho++) {
                        for (uint32_t wo = 0; wo < W_out; wo++) {
                            // Pad conditions
                            int pad_cond_h = 2 * ho + hk - 1;
                            int pad_cond_w = 2 * wo + wk - 1;

                            if ((pad_cond_h >= 0) && (pad_cond_w >= 0) && (pad_cond_h < H_in) && (pad_cond_w < W_in)) {
                                int out_idx = wo + ho * W_out + co * H_out * W_out;
                                int in_idx = (2 * wo + wk - 1) + (2 * ho + hk - 1) * W_in + ci * H_in * W_in;

                                temp += outDiff[out_idx] * inData[in_idx];

                                if (USE_BIASES == 1) bias_temp += outDiff[out_idx];
                            }
                        }
                    }
                    coeffDiff[wk + hk * 5 + ci * 25 + co * 25 * C_in] = temp;

                    if (USE_BIASES == 1) {
                        biasDiff[co] = bias_temp;
                    }
                }
            }
        }
    }

    if (USE_BIASES != 0 && USE_BIASES != 1) {
        printf("[naive_conv2d_param_grad_kernel_CHW_k5x5_s2_p1:] Invalid selection of the bias option (1 or 0 - use biases or not). Actual value: %d. Biases not used, even if provided!\n",
               USE_BIASES);
    }
}


void naive_conv2d_in_grad_kernel_CHW_k5x5_s2_p1(void *matMul_args) {
    struct matMul_args *args = (struct matMul_args *) matMul_args;
    float *__restrict__ inDiff = args->A;
    float *__restrict__ coeffData = args->B;
    float *__restrict__ outDiff = args->C;

    const uint32_t USE_BIASES = args->USE_BIASES;

    const uint32_t H_in = args->H;
    const uint32_t W_in = args->W;
    const uint32_t C_in = args->pCin;

    const uint32_t H_out = (H_in - 3) / 2 + 1;
    const uint32_t W_out = (W_in - 3) / 2 + 1;
    const uint32_t C_out = args->pCout;

    const uint32_t pHW = 24;

    const uint32_t blockSize = (C_in + NUM_CORES - 1) / NUM_CORES;
    const uint32_t start = pi_core_id() * blockSize;
    const uint32_t stop = start + blockSize > C_in ? C_in : start + blockSize;

    int h_start = 3;
    int w_start = 3;

    for (uint32_t ci = start; ci < stop; ci++) {
        for (uint32_t hi = 0; hi < H_in; hi++) {
            for (uint32_t wi = 0; wi < W_in; wi++) {
                float temp = 0;

                for (uint32_t co = 0; co < C_out; co++) {
                    for (uint32_t hk = 0; hk < 5; hk++) {
                        for (uint32_t wk = 0; wk < 5; wk++) {
                            // Indices
                            int ker_idx = (pHW - wk - hk * 5) + ci * 25 + co * 25 * C_in;

                            // Border conditions
                            int bord_h = (hi + hk - h_start) / 2;
                            int bord_w = (wi + wk - w_start) / 2;
                            int borders = (bord_h < H_out) && (bord_w < W_out);

                            float o_grad = 0;
                            float k_dat = coeffData[ker_idx];

                            if (((hi + hk - h_start) % 2 == 0) && ((wi + wk - w_start) % 2 == 0) && borders == 1) {
                                int out_idx = bord_w + (bord_h) * W_out + co * H_out * W_out;

                                o_grad = outDiff[out_idx];
                                temp += k_dat * o_grad;
                            }
                        }
                    }
                }

                inDiff[wi + hi * W_in + ci * H_in * W_in] = temp;
            }
        }
    }

    if (USE_BIASES != 0 && USE_BIASES != 1) {
        printf("[naive_conv2d_in_grad_kernel_CHW_k5x5_s2_p2:] Invalid selection of the bias option (1 or 0 - use biases or not). Actual value: %d. Current step not affected by this.\n",
               USE_BIASES);
    }
}








/** TRANSPOSED CONV2D KERNELS **/

void naive_transp_conv2d_fw_kernel_CHW (void * matMul_args) 
{
  struct matMul_args* args = (struct matMul_args *)matMul_args;
  float * __restrict__ inData = args->A;
  float * __restrict__ coeffData = args->B;
  float * __restrict__ outData = args->C;

  float *__restrict__ biasData = args->bias;
  const uint32_t USE_BIASES = args->USE_BIASES;

  const uint32_t H_in = args->H;
  const uint32_t W_in = args->W;
  const uint32_t pW = args->pW;
  const uint32_t pH = args->pH;
  const uint32_t C_in = args->pCin;
  const uint32_t C_out = args->pCout;

  uint32_t h_str = args->stride_h;
  uint32_t w_str = args->stride_w;
  uint32_t Lpad = args->Lpad;
  uint32_t Rpad = args->Rpad;
  uint32_t Upad = args->Upad;
  uint32_t Dpad = args->Dpad;

  const uint32_t H_out = (H_in - 1) * h_str - (Upad + Dpad) + (pH - 1) + 1;
  const uint32_t W_out = (W_in - 1) * w_str - (Lpad + Rpad) + (pW - 1) + 1;

  const uint32_t blockSize = (C_out+NUM_CORES-1) / NUM_CORES;
  const uint32_t start = pi_core_id()*blockSize;
  const uint32_t stop = start+blockSize > C_out ? C_out : start+blockSize;  

  int padding = Lpad + Rpad + Upad + Dpad;

  if (USE_BIASES == 1) {
    // Initialize the output with bias term for each output channel
    for (int co = start; co < stop; ++co) {
      for (int ho = 0; ho < H_out; ++ho) {
        for (int wo = 0; wo < W_out; ++wo) {
          outData[wo + ho*W_out + co*H_out*W_out] = biasData[co];
        }
      }
    }
  }

  // Perform the transposed convolution
  for (int hi = 0; hi < H_in; ++hi) {
    for (int wi = 0; wi < W_in; ++wi) {
      for (int hk = 0; hk < pH; ++hk) {
        for (int wk = 0; wk < pW; ++wk) {
          int out_i = hi * h_str + hk - Upad;
          int out_j = wi * w_str + wk - Lpad;
          for (int co = start; co < stop; ++co) {
            float temp = 0;
            for (int ci = 0; ci < C_in; ++ci) {
              if (out_i >= 0 && out_i < H_out && out_j >= 0 && out_j < W_out) {
                //outData[out_j + out_i*W_out + co*H_out*W_out] += inData[wi + hi*W_in + ci*H_in*W_in] * coeffData[wk + hk*pW + ci*pH*pW + co*C_in*pH*pW];
                temp += inData[wi + hi*W_in + ci*H_in*W_in] * coeffData[wk + hk*pW + ci*pH*pW + co*C_in*pH*pW];
              }
            }
            outData[out_j + out_i*W_out + co*H_out*W_out] += temp;
          }
        }
      }
    }
  }

  if (USE_BIASES != 0 && USE_BIASES != 1) {
      printf("[naive_transp_conv2d_fw_kernel_CHW:] Invalid selection of the bias option (1 or 0 - use biases or not). Actual value: %d. Biases not used, even if provided!\n",
             USE_BIASES);
  }
}



void naive_transp_conv2d_param_grad_kernel_CHW (void * matMul_args) 
{
  struct matMul_args* args = (struct matMul_args *)matMul_args;
  float * __restrict__ inData = args->A;
  float * __restrict__ coeffDiff = args->B;
  float * __restrict__ outDiff = args->C;

  float *__restrict__ biasDiff = args->bias;
  const uint32_t USE_BIASES = args->USE_BIASES;

  const uint32_t H_in = args->H;
  const uint32_t W_in = args->W;
  const uint32_t pW = args->pW;
  const uint32_t pH = args->pH;
  const uint32_t C_in = args->pCin;
  const uint32_t C_out = args->pCout;

  uint32_t h_str = args->stride_h;
  uint32_t w_str = args->stride_w;
  uint32_t Lpad = args->Lpad;
  uint32_t Rpad = args->Rpad;
  uint32_t Upad = args->Upad;
  uint32_t Dpad = args->Dpad;

  const uint32_t H_out = (H_in - 1) * h_str - (Upad + Dpad) + (pH - 1) + 1;
  const uint32_t W_out = (W_in - 1) * w_str - (Lpad + Rpad) + (pW - 1) + 1;

  const uint32_t blockSize = (C_out+NUM_CORES-1) / NUM_CORES;
  const uint32_t start = pi_core_id()*blockSize;
  const uint32_t stop = start+blockSize > C_out ? C_out : start+blockSize;  

  int padding = Lpad + Rpad + Upad + Dpad;

  for (uint32_t co = start; co < stop; ++co) {
    for (uint32_t hk = 0; hk < pH; ++hk) {
      for (uint32_t wk = 0; wk < pW; ++wk) {
        for (uint32_t ci = 0; ci < C_in; ++ci) {
          float temp = 0;
          for (uint32_t hi = 0; hi < H_in; ++hi) {
            for (uint32_t wi = 0; wi < W_in; ++wi) {
              int out_i = hi * h_str + hk - Upad;
              int out_j = wi * w_str + wk - Lpad;
              if (out_i >= 0 && out_i < H_out && out_j >= 0 && out_j < W_out) {
                //coeffDiff[wk + hk*pW + co*pW*pH + ci*C_out*pW*pH] += inData[wi + hi*W_in + ci*H_in*W_in] * outDiff[out_j + out_i*W_out + co*H_out*W_out];
                temp += inData[wi + hi*W_in + ci*H_in*W_in] * outDiff[out_j + out_i*W_out + co*H_out*W_out];
              }
            }
          }
          coeffDiff[wk + hk*pW + co*pW*pH + ci*C_out*pW*pH] = temp;
        }
      }
    }
  }

  if (USE_BIASES == 1) {
    // Compute the bias gradient as the sum of dY for each output channel
    for (int c_o = start; c_o < stop; ++c_o) {
      float temp = 0;
      for (int ho = 0; ho < H_out; ++ho) {
        for (int wo = 0; wo < W_out; ++wo) {
          //biasDiff[c_o] += outDiff[wo + ho*W_out + c_o*H_out*W_out];
          temp += outDiff[wo + ho*W_out + c_o*H_out*W_out];
        }
      }
      biasDiff[c_o] = temp;
    }
  }

  if (USE_BIASES != 0 && USE_BIASES != 1) {
      printf("[naive_transp_conv2d_param_grad_kernel_CHW:] Invalid selection of the bias option (1 or 0 - use biases or not). Actual value: %d. Biases not used, even if provided!\n",
             USE_BIASES);
  }
}



void naive_transp_conv2d_in_grad_kernel_CHW (void * matMul_args) 
{
  struct matMul_args* args = (struct matMul_args *)matMul_args;
  float * __restrict__ inDiff = args->A;
  float * __restrict__ coeffData = args->B;
  float * __restrict__ outDiff = args->C;

  const uint32_t USE_BIASES = args->USE_BIASES;

  const uint32_t H_in = args->H;
  const uint32_t W_in = args->W;
  const uint32_t pW = args->pW;
  const uint32_t pH = args->pH;
  const uint32_t C_in = args->pCin;
  const uint32_t C_out = args->pCout;

  uint32_t h_str = args->stride_h;
  uint32_t w_str = args->stride_w;
  uint32_t Lpad = args->Lpad;
  uint32_t Rpad = args->Rpad;
  uint32_t Upad = args->Upad;
  uint32_t Dpad = args->Dpad;

  const uint32_t H_out = (H_in - 1) * h_str - (Upad + Dpad) + (pH - 1) + 1;
  const uint32_t W_out = (W_in - 1) * w_str - (Lpad + Rpad) + (pW - 1) + 1;
  const uint32_t pHW = pH*pW-1;

  const uint32_t blockSize = (C_in+NUM_CORES-1) / NUM_CORES;
  const uint32_t start = pi_core_id()*blockSize;
  const uint32_t stop = start+blockSize > C_in ? C_in : start+blockSize;  

  // Compute the input gradient
  for (int hi = 0; hi < H_in; ++hi) {
    for (int wi = 0; wi < W_in; ++wi) {
      for (int hk = 0; hk < pH; ++hk) {
        for (int wk = 0; wk < pW; ++wk) {
          int out_i = hi * h_str + hk - Upad;
          int out_j = wi * w_str + wk - Lpad; 
          for (int ci = start; ci < stop; ++ci) {
            float temp = 0;
            for (int co = 0; co < C_out; ++co) {
              if (out_i >= 0 && out_i < H_out && out_j >= 0 && out_j < W_out) {
                //inDiff[wi + hi*W_in + ci*H_in*W_in] += coeffData[wk + hk*pW + co*pH*pW + ci*pW*pH*C_out] * outDiff[out_j + out_i*W_out + co*H_out*W_out];
                temp += coeffData[wk + hk*pW + co*pH*pW + ci*pW*pH*C_out] * outDiff[out_j + out_i*W_out + co*H_out*W_out];
              }
            }
            inDiff[wi + hi*W_in + ci*H_in*W_in] += temp;
          }
        }
      }
    }
  }

  if (USE_BIASES != 0 && USE_BIASES != 1) {
      printf("[naive_transp_conv2d_in_grad_kernel_CHW:] Invalid selection of the bias option (1 or 0 - use biases or not). Actual value: %d. Current step not affected by this.\n",
             USE_BIASES);
  }
}
