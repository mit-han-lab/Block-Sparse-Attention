# Adapted from https://github.com/Dao-AILab/flash-attention/blob/main/benchmarks/benchmark_flash_attention.py

import torch

from block_sparse_attn import (
    block_sparse_attn_func,
    flash_attn_varlen_func,
)

from utils import (
    time_fwd,
    flops,
    efficiency,
    write_to_excel,
)

def generate_base_sparsity_mask(max_seqlen_q, max_seqlen_k, round_base, m_block_dim, n_block_dim, sparsity, causal=False, device="cuda"):
    def round_to_multiple(x, base):
        return ((x + base - 1) // base) * base
    nrow, ncol = round_to_multiple(max_seqlen_q, round_base) // m_block_dim, round_to_multiple(max_seqlen_k, round_base) // n_block_dim
    base_mask = torch.zeros(1, nrow, ncol, device=device, dtype=torch.bool)
    total_block_num = 0

    density = 1.0 - sparsity
    if not density == 0.0 and not density == 1.0:
        for i in range(nrow): # do in reverse order
            idx = nrow - i - 1
            if causal:
                available_col_num = max(0, ncol - i)
                total_block_num += available_col_num
                num_one = max(1, int(density * available_col_num))
                base_mask[0][idx, torch.randperm(available_col_num)[:num_one]] = True
            else:
                available_col_num = ncol
                total_block_num += available_col_num
                num_one = max(1, int(density * available_col_num))
                base_mask[0][idx, torch.randperm(available_col_num)[:num_one]] = True
    elif density == 1.0:
        base_mask[0] = torch.ones_like(base_mask[0])
        total_block_num = nrow * ncol
    else:
        total_block_num = nrow * ncol
    
    calculated_block_num = base_mask.sum().item()
    real_sparsity = 1.0 - calculated_block_num / total_block_num
    return base_mask, real_sparsity

block_size = 128

def get_sparsity_list(sampling_steps, seqlen, causal):
    blockmask_element_num = (seqlen // block_size) ** 2 // (2 if causal else 1)
    stride = max(blockmask_element_num // sampling_steps, 1)
    actual_steps = (blockmask_element_num + stride - 1) // stride
    sparsity_list = []
    for i in range(actual_steps):
        sparse_rate = (1 + i * stride) / blockmask_element_num
        if sparse_rate > 0.95 or sparse_rate < 0.0:
            continue
        sparsity_list.append(sparse_rate)
    return sparsity_list
    
    
def profile_blocksparse_fwd():
    repeats = 15
    block_sparse_repeats = 3
    device = 'cuda:0'
    dtype = torch.float16
    causal = True
    batch_size = 8
    sparsity_sampling_steps = 20
    seqlen_vals = [1024,2048,4096,8192,16384,32768,65536]
    headdim = 128
    dim = 4096
    dropout_p = 0.0
    method = ("Block_Sparse_Flash2")
    time_f = {}
    speed_f = {}


    excel_label = ["batch_size", "seqlen", "actual_sparsity", "speed", "latency", "speedup", "base_speed", "base_latency"]
    excel_data = []
    excel_dir_path = "./excel/blocksparse/"
    excel_file_name = f"hdim{headdim}_nheads{dim // headdim}_bts{batch_size}_fwd"
        
    if causal:
        excel_file_name += "_causal"
    
    all_results = {}
    for seqlen in seqlen_vals:
        results = {}
        nheads = dim // headdim
        shape = (batch_size * seqlen, nheads, headdim)
        q = torch.randn(shape, device=device, dtype=dtype)
        k = torch.randn(shape, device=device, dtype=dtype)
        v = torch.randn(shape, device=device, dtype=dtype)
        cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32, device=device)
        base_f = time_fwd(flash_attn_varlen_func, q, k, v, cu_seqlens, cu_seqlens, seqlen, seqlen, dropout_p, None, causal, repeats=repeats, verbose=False)
        base_speed = efficiency(flops(batch_size, seqlen, headdim, nheads, causal, mode="fwd"), base_f)
        results["base"] = [[base_f], [base_speed]]
        sparsity_list = get_sparsity_list(sparsity_sampling_steps, seqlen, causal)
        print(f"sparsity_list: {sparsity_list}")
        for sparsity in sparsity_list:
            sum_sparsity, sum_speed, sum_latency = 0, 0, 0
            for _ in range(block_sparse_repeats):
                cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32, device=device)
                head_mask_type = torch.tensor([1] * nheads, device=device, dtype=torch.int32)
                base_blockmask, real_sparsity = generate_base_sparsity_mask(seqlen, seqlen, block_size, block_size, block_size, sparsity, causal = causal, device=device)
                base_blockmask = base_blockmask.unsqueeze(0).repeat(batch_size, nheads, 1, 1)
                config = (causal, headdim, nheads, batch_size, seqlen, sparsity, real_sparsity)
                f = time_fwd(block_sparse_attn_func, q, k, v, cu_seqlens, cu_seqlens, head_mask_type, None, base_blockmask, seqlen, seqlen, dropout_p, is_causal=causal, exact_streaming=False, repeats=repeats, verbose=False)
                time_f[config, method] = f
                print(f"### causal={causal}, headdim={headdim}, nheads = {nheads}, batch_size={batch_size}, seqlen={seqlen}, real_sparsity={real_sparsity} ###")
                speed_f[config, method] = efficiency(flops(batch_size, seqlen, headdim, nheads, causal, mode="fwd"), time_f[config, method])
                print(
                    f"{method}"
                    f"fwd: {speed_f[config, method]:.2f} TFLOPs/s, {(time_f[config, method]*1000):.2f} ms, "
                    f"fwd base: {base_speed:.2f} TFLOPs/s, {base_f*1000:.2f} ms"
                    ) 
                sum_sparsity += real_sparsity
                sum_speed += speed_f[config, method]
                sum_latency += time_f[config, method]
            
            avg_sparsity = sum_sparsity / block_sparse_repeats
            avg_speed = sum_speed / block_sparse_repeats
            avg_latency = sum_latency / block_sparse_repeats
            if avg_sparsity not in results:
                    results[avg_sparsity] = [[],[]]
            results[avg_sparsity][0].append(avg_latency)
            results[avg_sparsity][1].append(avg_speed)
            excel_data.append([batch_size, seqlen, avg_sparsity, avg_speed, avg_latency, avg_speed / base_speed, base_speed, base_f])
        
        for key in results.keys():
            avg_latency = sum(results[key][0]) / len(results[key][0])
            avg_speed = sum(results[key][1]) / len(results[key][1])
            results[key] = [avg_latency, avg_speed]
        all_results[seqlen] = results
    
    import json
    with open(f"all_results_{excel_file_name}.json", "w") as f:
        json.dump(all_results, f)
            
    write_to_excel(excel_label, excel_data, excel_dir_path, excel_file_name)

profile_blocksparse_fwd()