import torch
from block_sparse_attn import token_streaming_attn_func
from flash_attn import flash_attn_varlen_func
from utils import time_fwd, flops, efficiency, write_to_excel

def profile_exact_streaming_fwd():
    # Configuration
    repeats = 20
    block_sparse_repeats = 10
    device = 'cuda:0'
    dtype = torch.float16
    causal = True
    batch_size = 8
    sink_local_num = [64, 256]
    seqlen_vals = [4096, 8192, 16384, 32768, 65536]
    headdim_vals = [128]
    dim = 4096
    dropout_p = 0.0

    for headdim in headdim_vals:
        nheads = dim // headdim
        excel_path = "./excel/streaming/"
        excel_name = f"hdim{headdim}_nheads{nheads}_bts{batch_size}_sink{sink_local_num[0]}_local{sink_local_num[1]}_fwd"
        excel_labels = ["batch_size", "seqlen", "speed", "latency", "speedup", "base_speed", "base_latency"]
        excel_data = []

        print(f"\nConfiguration: headdim={headdim}, nheads={nheads}, batch_size={batch_size}, causal={causal}")
        print(f"{'SeqLen':<10} | {'Base Latency':<15} | {'Base TFLOPs':<15} | {'Avg Latency':<15} | {'Avg TFLOPs':<15} | {'Speedup':<10}")
        print("-" * 95)

        for seqlen in seqlen_vals:
            shape = (batch_size * seqlen, nheads, headdim)
            q = torch.randn(shape, device=device, dtype=dtype)
            k = torch.randn(shape, device=device, dtype=dtype)
            v = torch.randn(shape, device=device, dtype=dtype)
            cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32, device=device)

            # Benchmark Base (Flash Attention)
            base_f = time_fwd(
                flash_attn_varlen_func, q, k, v, cu_seqlens, cu_seqlens, 
                seqlen, seqlen, dropout_p, None, causal, repeats=repeats, verbose=False
            )
            base_flops_val = flops(batch_size, seqlen, headdim, nheads, causal, mode="fwd")
            base_speed = efficiency(base_flops_val, base_f)

            # Prepare for Streaming
            head_mask_type = torch.tensor([-1] * (nheads//2) + [0] * (nheads - nheads//2), device=device, dtype=torch.int32)
            streaming_info = torch.tensor(sink_local_num * nheads, device=device, dtype=torch.int32)
            
            # Benchmark Streaming
            total_latency = 0
            for _ in range(block_sparse_repeats):
                f = time_fwd(
                    token_streaming_attn_func, q, k, v, cu_seqlens, cu_seqlens, 
                    head_mask_type, streaming_info, seqlen, seqlen, repeats=repeats, verbose=False
                )
                total_latency += f

            avg_latency = total_latency / block_sparse_repeats
            avg_speed = efficiency(base_flops_val, avg_latency)
            speedup = base_f / avg_latency if avg_latency > 0 else 0.0

            print(f"{seqlen:<10} | {base_f*1000:>12.2f} ms | {base_speed:>12.2f} | {avg_latency*1000:>12.2f} ms | {avg_speed:>12.2f} | {speedup:>9.2f}x")

            excel_data.append([batch_size, seqlen, avg_speed, avg_latency, speedup, base_speed, base_f])
        
        write_to_excel(excel_labels, excel_data, excel_path, excel_name)
        print(f"\nResults saved to {excel_path}{excel_name}.xlsx")

if __name__ == "__main__":
    profile_exact_streaming_fwd()
