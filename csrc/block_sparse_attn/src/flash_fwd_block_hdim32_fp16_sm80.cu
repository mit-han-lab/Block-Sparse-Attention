// Copyright (c) 2023, Tri Dao.
// Adapted by Junxian Guo from https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/flash_fwd_hdim32_fp16_sm80.cu
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_block_<cutlass::half_t, 32>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_block_hdim32<cutlass::half_t>(params, stream);
}
