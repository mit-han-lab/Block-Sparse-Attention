# Adapted from https://github.com/Dao-AILab/flash-attention/blob/main/tests/test_flash_attn.py

import pytest
import torch
from einops import repeat
from block_sparse_attn import (
    block_sparse_attn_func,
)
from utils import (
    generate_random_padding_mask,
    generate_base_sparsity_mask,
    generate_qkv,
    generate_streaming_mask,
    prepare_mixed_exact_mask,
    prepare_mixed_mask,
    convert_flash_attn_S_to_softmax,
    normalize_flash_attn_S,
    get_dropout_fraction,
    attention_blocksparse_ref
)

MAX_HEADDIM_SM8x = 192
block_size = 128
is_sm75 = torch.cuda.get_device_capability("cuda") == (7, 5)
is_sm8x = torch.cuda.get_device_capability("cuda")[0] == 8
is_sm80 = torch.cuda.get_device_capability("cuda") == (8, 0)
is_sm90 = torch.cuda.get_device_capability("cuda") == (9, 0)


@pytest.mark.parametrize("dtype", ([torch.float16] if is_sm75 else [torch.float16, torch.bfloat16]))
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
@pytest.mark.parametrize("d", [32, 64, 128])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [   
        (113, 203),
        (128, 217),
        (113, 211),
        (108, 256),
        (256, 512),
        (512, 256),
        (1024, 1024),
        (1023, 1024),
        (1024, 1023),
        (2048, 2048),
    ],
)

@pytest.mark.parametrize(
    "causal, exact_streaming, sink_num, local_num", 
    [
        # (True, True, 1, 3),
        # (True, True, 64, 256),
        (True, False, 1, 3),
        (False, False, 1, 3),
    ]
)

@pytest.mark.parametrize("p_dropout", [0.17, 0.0])
@pytest.mark.parametrize("sparsity", [0, 0.1, 0.3, 0.7, 1.0])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("nheads", [16, 32])

def test_flash_attn_varlen_block_output(
    seqlen_q, seqlen_k, d, p_dropout, causal, exact_streaming, sink_num, local_num, mha_type, dtype, sparsity, batch_size, nheads
):
    if (
        max(seqlen_q, seqlen_k) >= 2048
        and torch.cuda.get_device_properties("cuda").total_memory <= 16 * 2**30
    ):
        pytest.skip()  # Reference implementation OOM
    device = "cuda:0"
    # set seed
    torch.random.manual_seed(42)
    nheads_k = nheads if mha_type == "mha" else (1 if mha_type == "mqa" else 8)
    assert nheads % nheads_k == 0
    window_size = (-1, -1)
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype, requires_grad=True)

    query_padding_mask = generate_random_padding_mask(seqlen_q, batch_size, device, mode="random")
    key_padding_mask = generate_random_padding_mask(seqlen_k, batch_size, device, mode="random")

    alibi_slopes, attn_bias = None, None
    (
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        q,
        k,
        v,
        output_pad_fn,
        dq_pad_fn,
        dk_pad_fn,
    ) = generate_qkv(q, k, v, query_padding_mask, key_padding_mask, kvpacked=False)
    
    num_streaming_heads = nheads // 3
    num_blocksparse_heads = nheads // 3
    num_dense_heads = nheads - num_streaming_heads - num_blocksparse_heads
    sparsity_list = [sparsity] * num_blocksparse_heads
    head_mask_type = torch.tensor([0] * num_dense_heads + [1] * num_blocksparse_heads + [-1] * num_streaming_heads, device=device, dtype=torch.int32)
    base_blockmask = generate_base_sparsity_mask(max_seqlen_q, max_seqlen_k, block_size, block_size, block_size, batch_size, num_blocksparse_heads, sparsity_list, causal = causal, device=device)
    
    streaming_info = torch.tensor([sink_num, local_num] * nheads, device=device, dtype=torch.int32)
    streaming_mask = generate_streaming_mask(max_seqlen_q, max_seqlen_k, batch_size, nheads, cu_seqlens_q, cu_seqlens_k, block_size, block_size, block_size, streaming_info, causal=causal, device=device)
    
    if exact_streaming:
        assert causal
    print(f"exact_streaming: {exact_streaming}")
    if exact_streaming:
        mixed_mask = prepare_mixed_exact_mask(base_blockmask, streaming_info, head_mask_type, batch_size, nheads, block_size, block_size, block_size, max_seqlen_q, max_seqlen_k, q.shape[1], k.shape[1], query_padding_mask, key_padding_mask, device=device)
    else:
        mixed_mask = prepare_mixed_mask(base_blockmask, streaming_mask, head_mask_type, batch_size, nheads, block_size, block_size, block_size, max_seqlen_q, max_seqlen_k, q.shape[1], k.shape[1], device=device)
    
    
    out_unpad, sm_lse, S_dmask = block_sparse_attn_func(
        q_unpad, k_unpad, v_unpad,
        cu_seqlens_q, cu_seqlens_k,
        head_mask_type,
        streaming_info,
        base_blockmask,
        max_seqlen_q, max_seqlen_k,
        p_dropout,
        deterministic=True,
        softmax_scale=None,
        is_causal=causal,
        exact_streaming=exact_streaming,
        return_attn_probs=True,
    )
    
    out = output_pad_fn(out_unpad)
    
    if p_dropout > 0.0:
        assert S_dmask is not None
        S_dmask_converted = convert_flash_attn_S_to_softmax(
            S_dmask,
            seqlen_q,
            seqlen_k,
            query_padding_mask,
            key_padding_mask,
            d,
            p_dropout > 0.0,
            causal=causal,
            window_size=window_size,
        )
        dropout_mask = S_dmask_converted >= 0
        attn_unnorm = S_dmask_converted.abs()
        
        k_rep = repeat(k, "b s h d -> b s (h g) d", g=nheads // nheads_k)
        v_rep = repeat(v, "b s h d -> b s (h g) d", g=nheads // nheads_k)
        
        attn = normalize_flash_attn_S(
            attn_unnorm,
            q,
            k_rep,
            v_rep,
            query_padding_mask,
            key_padding_mask,
            attn_bias,
            p_dropout > 0.0,
            causal=causal,
            window_size=window_size,
        )
        
        dropout_fraction = get_dropout_fraction(
            dropout_mask,
            mixed_mask,
            block_size, block_size, 
            query_padding_mask,
            key_padding_mask,
            causal=causal,
            window_size=window_size,
        ).item()
        
        print(f"Actual dropout fraction: {dropout_fraction}")
    else:
        dropout_mask = None

    out_ref, attn_ref = attention_blocksparse_ref(
            q,
            k,
            v,
            mixed_mask,
            block_size, block_size, 
            query_padding_mask,
            key_padding_mask,
            p_dropout,
            dropout_mask,
            causal=causal,
            window_size=window_size,
        )
    out_pt, attn_pt = attention_blocksparse_ref(
            q,
            k,
            v,
            mixed_mask,
            block_size, block_size, 
            query_padding_mask,
            key_padding_mask,
            p_dropout,
            dropout_mask,
            causal=causal,
            window_size=window_size,
            upcast=False,
            reorder_ops=True,
        )

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")

    g = torch.randn_like(out)
    # g = torch.zeros_like(out)
    if d <= MAX_HEADDIM_SM8x or (is_sm80 or is_sm90):
        (
            dq_unpad,
            dk_unpad,
            dv_unpad,
        ) = torch.autograd.grad(out, (q_unpad, k_unpad, v_unpad), g)
        dk = dk_pad_fn(dk_unpad)
        dv = dk_pad_fn(dv_unpad)
        (
            dq_ref,
            dk_ref,
            dv_ref,
        ) = torch.autograd.grad(out_ref, (q, k, v), g)
        (
            dq_pt,
            dk_pt,
            dv_pt,
        ) = torch.autograd.grad(out_pt, (q, k, v), g)
        dq = dq_pad_fn(dq_unpad)
        
        
        print(f"dQ max diff: {(dq - dq_ref).abs().max().item()}")
        print(f"dK max diff: {(dk - dk_ref).abs().max().item()}")
        print(f"dV max diff: {(dv - dv_ref).abs().max().item()}")
        print(f"dQ mean diff: {(dq - dq_ref).abs().mean().item()}")
        print(f"dK mean diff: {(dk - dk_ref).abs().mean().item()}")
        print(f"dV mean diff: {(dv - dv_ref).abs().mean().item()}")
        print(f"dQ Pytorch max diff: {(dq_pt - dq_ref).abs().max().item()}")
        print(f"dK Pytorch max diff: {(dk_pt - dk_ref).abs().max().item()}")
        print(f"dV Pytorch max diff: {(dv_pt - dv_ref).abs().max().item()}")
        print(f"dQ Pytorch mean diff: {(dq_pt - dq_ref).abs().mean().item()}")
        print(f"dK Pytorch mean diff: {(dk_pt - dk_ref).abs().mean().item()}")
        print(f"dV Pytorch mean diff: {(dv_pt - dv_ref).abs().mean().item()}")
        
        
    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item()

    if d <= MAX_HEADDIM_SM8x or (is_sm80 or is_sm90):
        assert (dq - dq_ref).abs().max().item() <= 3 * (dq_pt - dq_ref).abs().max().item()
        assert (dk - dk_ref).abs().max().item() <= 3 * (dk_pt - dk_ref).abs().max().item()
        assert (dv - dv_ref).abs().max().item() <= 3 * (dv_pt - dv_ref).abs().max().item()