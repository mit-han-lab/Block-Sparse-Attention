import torch
from block_sparse_attn import block_streaming_attn_func
from flash_attn import flash_attn_varlen_func
from utils import time_fwd_bwd, flops, efficiency, write_to_excel

def profile_block_streaming_fwd_bwd():
    # Configuration
    repeats = 10
    block_sparse_repeats = 5
    device = 'cuda:0'
    dtype = torch.float16
    causal = True
    batch_size = 1
    sink_local_block_num = [1, 3]
    seqlen_vals = [1024, 2048, 4096, 8192, 16384, 20480, 24576, 28672, 32768, 65536, 131072]
    headdim_vals = [128]
    dim = 4096
    p_dropout = 0.0

    for headdim in headdim_vals:
        nheads = dim // headdim
        excel_path = "./excel/block_streaming/"
        excel_name = f"hdim{headdim}_nheads{nheads}_bts{batch_size}_sink_block{sink_local_block_num[0]}_local_block{sink_local_block_num[1]}_fwd_bwd"
        excel_labels = ["batch_size", "seqlen", "speed", "latency", "speedup", "base_speed", "base_latency"]
        excel_data = []

        print(f"\nConfiguration: headdim={headdim}, nheads={nheads}, batch_size={batch_size}, causal={causal}")
        print(f"{'SeqLen':<10} | {'Base Latency':<15} | {'Base TFLOPs':<15} | {'Avg Latency':<15} | {'Avg TFLOPs':<15} | {'Speedup':<10}")
        print("-" * 95)

        for seqlen in seqlen_vals:
            shape = (batch_size * seqlen, nheads, headdim)
            q = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
            k = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
            v = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
            cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32, device=device)

            # Benchmark Base (Flash Attention)
            base_f, base_b = time_fwd_bwd(
                flash_attn_varlen_func, q, k, v, cu_seqlens, cu_seqlens, 
                seqlen, seqlen, p_dropout, None, causal, repeats=repeats, verbose=False
            )
            base_latency = base_f + base_b
            base_flops_val = flops(batch_size, seqlen, headdim, nheads, causal, mode="fwd_bwd")
            base_speed = efficiency(base_flops_val, base_latency)

            # Prepare for Block Streaming
            head_mask_type = torch.tensor([-1] * (nheads//2) + [0] * (nheads - nheads//2), device=device, dtype=torch.int32)
            streaming_info = torch.tensor(sink_local_block_num * nheads, device=device, dtype=torch.int32)
            
            # Benchmark Block Streaming
            total_latency = 0
            for _ in range(block_sparse_repeats):
                f, b = time_fwd_bwd(
                    block_streaming_attn_func, q, k, v, cu_seqlens, cu_seqlens, 
                    head_mask_type, streaming_info, seqlen, seqlen, p_dropout, 
                    False, None, causal, repeats=repeats, verbose=False
                )
                total_latency += (f + b)

            avg_latency = total_latency / block_sparse_repeats
            avg_speed = efficiency(base_flops_val, avg_latency)
            speedup = base_latency / avg_latency if avg_latency > 0 else 0.0

            print(f"{seqlen:<10} | {base_latency*1000:>12.2f} ms | {base_speed:>12.2f} | {avg_latency*1000:>12.2f} ms | {avg_speed:>12.2f} | {speedup:>9.2f}x")

            excel_data.append([batch_size, seqlen, avg_speed, avg_latency, speedup, base_speed, base_latency])
        
        write_to_excel(excel_labels, excel_data, excel_path, excel_name)

if __name__ == "__main__":
    profile_block_streaming_fwd_bwd()
