# Adapted from https://github.com/Dao-AILab/flash-attention/blob/main/benchmarks/benchmark_flash_attention.py

import openpyxl
from block_sparse_attn.utils.benchmark import benchmark_forward
import math
import torch

from block_sparse_attn import (
    token_streaming_attn_func,
    flash_attn_varlen_func,
)

from utils import (
    time_fwd,
    flops,
    efficiency,
    write_to_excel,
)


def profile_exact_streaming_fwd():
    repeats = 20
    block_sparse_repeats = 10
    device = 'cuda:0'
    dtype = torch.float16
    causal = True
    batch_size = 8
    sink_local_num = [64,256]
    seqlen_vals = [4096,8192,16384,32768,65536]
    headdim_vals = [128]
    dim = 4096
    dropout_p = 0.0
    methods = (["Flash2"])
    time_f = {}
    speed_f = {}

    for headdim in headdim_vals:
        excel_label = ["batch_size", "seqlen", "speed", "latency", "speedup", "base_speed", "base_latency"]
        excel_data = []
        excel_dir_path = "./excel/streaming/"
        excel_file_name = f"hdim{headdim}_nheads{dim // headdim}_bts{batch_size}_sink{sink_local_num[0]}_local{sink_local_num[1]}_fwd"

        for seqlen in seqlen_vals:
            nheads = dim // headdim
            shape = (batch_size * seqlen, nheads, headdim)
            q = torch.randn(shape, device=device, dtype=dtype)
            k = torch.randn(shape, device=device, dtype=dtype)
            v = torch.randn(shape, device=device, dtype=dtype)
            cu_seqlens = torch.arange(
                0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32, device=device)
            base_f = time_fwd(flash_attn_varlen_func, q, k, v, cu_seqlens, cu_seqlens, seqlen, seqlen, dropout_p, None, causal, repeats=repeats, verbose=False)
            base_speed = efficiency(flops(batch_size, seqlen, headdim, nheads, causal, mode="fwd"), base_f)
            head_mask_type = torch.tensor([-1] * (nheads//2) + [0] * (nheads - nheads//2), device=device, dtype=torch.int32)
            streaming_info = torch.tensor([sink_local_num[0], sink_local_num[1]] * nheads, device=device, dtype=torch.int32)
            config = (causal, headdim, nheads, batch_size, seqlen, sink_local_num[0], sink_local_num[1])
            sum_speed, sum_latency = 0,0
            for _ in range(block_sparse_repeats):
                f = time_fwd(
                    token_streaming_attn_func, q, k, v, cu_seqlens, cu_seqlens, head_mask_type, streaming_info, seqlen, seqlen, repeats=repeats, verbose=False
                )
                time_f[config, "Flash2"] = f
                print(f"### causal={causal}, headdim={headdim}, nheads = {nheads}, batch_size={batch_size}, seqlen={seqlen}, sink={sink_local_num[0]}, local={sink_local_num[1]} ###")
                for method in methods:
                    speed_f[config, method] = efficiency(
                        flops(batch_size, seqlen, headdim,
                                nheads, causal, mode="fwd"),
                        time_f[config, method]
                    )
                    print(f"{method} fwd: {speed_f[config, method]:.2f} TFLOPs/s, {(time_f[config, method]*1000):.2f} ms, ")   
                sum_speed += speed_f[config, "Flash2"]
                sum_latency += time_f[config, "Flash2"]
            excel_data.append([batch_size, seqlen, sum_speed / block_sparse_repeats, sum_latency / block_sparse_repeats, (sum_speed / block_sparse_repeats) / base_speed, base_speed, base_f])
        write_to_excel(excel_label, excel_data, excel_dir_path, excel_file_name)

profile_exact_streaming_fwd()