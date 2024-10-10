# Adapted from https://github.com/Dao-AILab/flash-attention/blob/main/benchmarks/benchmark_flash_attention.py

import openpyxl
from block_sparse_attn.utils.benchmark import benchmark_forward
import math
import torch

from block_sparse_attn import (
    block_streaming_attn_func,
    flash_attn_varlen_func,
)

from utils import (
    time_fwd_bwd,
    flops,
    efficiency,
    write_to_excel,
)


def profile_block_streaming_fwd_bwd():
    repeats = 10
    block_sparse_repeats = 5
    device = 'cuda:0'
    dtype = torch.float16
    causal = True
    batch_size = 1
    sink_local_block_num = [1,3]
    seqlen_vals = [1024, 2048, 4096, 8192, 16384, 20480, 24576, 28672, 32768, 65536, 131072]
    headdim_vals = [128]
    dim = 4096
    p_dropout = 0.0
    methods = (["Flash2"])
    time_f = {}
    time_b = {}
    time_f_b = {}
    speed_f = {}
    speed_b = {}
    speed_f_b = {}

    for headdim in headdim_vals:
        excel_label = ["batch_size", "seqlen", "speed", "latency", "speedup", "base_speed", "base_latency"]
        excel_data = []
        excel_dir_path = "./excel/block_streaming/"
        excel_file_name = f"hdim{headdim}_nheads{dim // headdim}_bts{batch_size}_sink_block{sink_local_block_num[0]}_local_block{sink_local_block_num[1]}_fwd_bwd"

        for seqlen in seqlen_vals:
            nheads = dim // headdim
            shape = (batch_size * seqlen, nheads, headdim)
            q = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
            k = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
            v = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
            cu_seqlens = torch.arange(
                0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32, device=device)
            base_f, base_b = time_fwd_bwd(flash_attn_varlen_func, q, k, v, cu_seqlens, cu_seqlens, seqlen, seqlen, p_dropout, None, causal, repeats=repeats, verbose=False)
            base_speed = efficiency(flops(batch_size, seqlen, headdim, nheads, causal, mode="fwd_bwd"), base_f + base_b)
            head_mask_type = torch.tensor([-1] * (nheads//2) + [0] * (nheads - nheads//2), device=device, dtype=torch.int32)
            streaming_info = torch.tensor([sink_local_block_num[0], sink_local_block_num[1]] * nheads, device=device, dtype=torch.int32)
            config = (causal, headdim, nheads, batch_size, seqlen, sink_local_block_num[0], sink_local_block_num[1])
            sum_speed, sum_latency = 0,0
            for _ in range(block_sparse_repeats):
                f, b = time_fwd_bwd(
                    block_streaming_attn_func, q, k, v, cu_seqlens, cu_seqlens, head_mask_type, streaming_info, seqlen, seqlen, p_dropout, False, None, causal, repeats=repeats, verbose=False
                )
                time_f[config, "Flash2"] = f
                time_b[config, "Flash2"] = b
                print(f"### causal={causal}, headdim={headdim}, nheads = {nheads}, batch_size={batch_size}, seqlen={seqlen}, sink={sink_local_block_num[0]}, local={sink_local_block_num[1]} ###")
                for method in methods:
                    time_f_b[config, method] = time_f[config, method] + time_b[config, method]
                    speed_f[config, method] = efficiency(
                        flops(batch_size, seqlen, headdim, nheads, causal, mode="fwd"),
                        time_f[config, method]
                    )
                    speed_b[config, method] = efficiency(
                        flops(batch_size, seqlen, headdim, nheads, causal, mode="bwd"),
                        time_b[config, method]
                    )
                    speed_f_b[config, method] = efficiency(
                        flops(batch_size, seqlen, headdim, nheads, causal, mode="fwd_bwd"),
                        time_f_b[config, method]
                    )
                    print(
                        f"{method}"
                        f"fwd: {speed_f[config, method]:.2f} TFLOPs/s, {(time_f[config, method]*1000):.2f} ms, "
                        f"bwd: {speed_b[config, method]:.2f} TFLOPs/s, {(time_b[config, method]*1000):.2f} ms, "
                        f"fwd + bwd: {speed_f_b[config, method]:.2f} TFLOPs/s, {(time_f_b[config, method]*1000):.2f} ms, "
                        f"fwd + bwd base: {base_speed:.2f} TFLOPs/s, {(base_f + base_b)*1000:.2f} ms"
                        )   
                sum_speed += speed_f_b[config, "Flash2"]
                sum_latency += time_f_b[config, "Flash2"]
            excel_data.append([batch_size, seqlen, sum_speed / block_sparse_repeats, sum_latency / block_sparse_repeats, (sum_speed / block_sparse_repeats) / base_speed, base_speed, base_f])
        write_to_excel(excel_label, excel_data, excel_dir_path, excel_file_name)

profile_block_streaming_fwd_bwd()