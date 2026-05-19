import triton
import triton.language as tl
import torch
from typing import Optional, Tuple

@triton.jit
def _block_sparse_attn_fwd_kernel(
    Q, K, V, Out, LSE,
    cu_seqlens_q, cu_seqlens_k,
    head_mask_type,
    streaming_info,
    blockmask,
    softmax_scale,
    stride_q_tok, stride_q_head, stride_q_dim,
    stride_k_tok, stride_k_head, stride_k_dim,
    stride_v_tok, stride_v_head, stride_v_dim,
    stride_o_tok, stride_o_head, stride_o_dim,
    max_seqlen_q, max_seqlen_k,
    num_heads, num_blocksparse_heads, q_slices, k_slices,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    HQ_OVER_HKV: tl.constexpr,
    EXACT_STREAMING: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    m_idx = tl.program_id(2)
    
    start_q = tl.load(cu_seqlens_q + batch_idx)
    end_q = tl.load(cu_seqlens_q + batch_idx + 1)
    seqlen_q = end_q - start_q
    
    start_k = tl.load(cu_seqlens_k + batch_idx)
    end_k = tl.load(cu_seqlens_k + batch_idx + 1)
    seqlen_k = end_k - start_k
    
    if m_idx * BLOCK_M >= seqlen_q:
        return
        
    kv_head_idx = head_idx // HQ_OVER_HKV
    
    q_offset = (start_q + m_idx * BLOCK_M + tl.arange(0, BLOCK_M))[:, None] * stride_q_tok + head_idx * stride_q_head + tl.arange(0, BLOCK_DMODEL)[None, :] * stride_q_dim
    q_mask = (start_q + m_idx * BLOCK_M + tl.arange(0, BLOCK_M))[:, None] < end_q
    q_chunk = tl.load(Q + q_offset, mask=q_mask, other=0.0)
    
    m_type = tl.load(head_mask_type + head_idx)
    
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - 1e38
    d_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    o_i = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    # ------------------ Dense Attention ------------------
    if m_type == 0:
        max_k_blocks = tl.cdiv(seqlen_k, BLOCK_N)
        for k_blk in range(0, max_k_blocks):
            k_start = k_blk * BLOCK_N
            k_offset = (start_k + k_start + tl.arange(0, BLOCK_N))[None, :] * stride_k_tok + kv_head_idx * stride_k_head + tl.arange(0, BLOCK_DMODEL)[:, None] * stride_k_dim
            v_offset = (start_k + k_start + tl.arange(0, BLOCK_N))[:, None] * stride_v_tok + kv_head_idx * stride_v_head + tl.arange(0, BLOCK_DMODEL)[None, :] * stride_v_dim
            
            k_chunk = tl.load(K + k_offset, mask=(start_k + k_start + tl.arange(0, BLOCK_N))[None, :] < end_k, other=0.0)
            v_chunk = tl.load(V + v_offset, mask=(start_k + k_start + tl.arange(0, BLOCK_N))[:, None] < end_k, other=0.0)
            
            qk = tl.dot(q_chunk, k_chunk) * softmax_scale
            q_rel = (start_q + m_idx * BLOCK_M + tl.arange(0, BLOCK_M))[:, None] - start_q
            k_rel = (start_k + k_start + tl.arange(0, BLOCK_N))[None, :] - start_k
            is_valid_qk = (q_rel + (seqlen_k - seqlen_q) >= k_rel) & (k_rel < seqlen_k)
            qk = tl.where(is_valid_qk, qk, -1e38)
            
            m_new = tl.max(qk, axis=1)
            m_next = tl.maximum(m_i, m_new)
            alpha = tl.math.exp(m_i - m_next)
            beta = tl.math.exp(qk - m_next[:, None])
            
            o_i = o_i * alpha[:, None]
            o_i = tl.dot(beta.to(v_chunk.dtype), v_chunk, o_i)
            
            d_new = tl.sum(beta, axis=1)
            d_i = d_i * alpha + d_new
            m_i = m_next
            
    # ------------------ Streaming Attention ------------------
    elif m_type < 0:
        sink_num = tl.load(streaming_info + head_idx * 2)
        local_num = tl.load(streaming_info + head_idx * 2 + 1)
        
        if EXACT_STREAMING:
            sink_blocks_cdiv = tl.cdiv(sink_num, BLOCK_N)
            local_blocks_cdiv = tl.cdiv(local_num, BLOCK_N)
            # Sink blocks
            max_sink_blocks = tl.minimum(sink_blocks_cdiv, tl.cdiv(seqlen_k, BLOCK_N))
        else:
            sink_blocks_cdiv = sink_num
            local_blocks_cdiv = local_num
            # Sink blocks
            max_sink_blocks = tl.minimum(sink_blocks_cdiv, tl.cdiv(seqlen_k, BLOCK_N))
            
        for k_blk in range(0, max_sink_blocks):
            k_start = k_blk * BLOCK_N
            k_offset = (start_k + k_start + tl.arange(0, BLOCK_N))[None, :] * stride_k_tok + kv_head_idx * stride_k_head + tl.arange(0, BLOCK_DMODEL)[:, None] * stride_k_dim
            v_offset = (start_k + k_start + tl.arange(0, BLOCK_N))[:, None] * stride_v_tok + kv_head_idx * stride_v_head + tl.arange(0, BLOCK_DMODEL)[None, :] * stride_v_dim
            k_chunk = tl.load(K + k_offset, mask=(start_k + k_start + tl.arange(0, BLOCK_N))[None, :] < end_k, other=0.0)
            v_chunk = tl.load(V + v_offset, mask=(start_k + k_start + tl.arange(0, BLOCK_N))[:, None] < end_k, other=0.0)
            
            qk = tl.dot(q_chunk, k_chunk) * softmax_scale
            if EXACT_STREAMING:
                q_rel = (start_q + m_idx * BLOCK_M + tl.arange(0, BLOCK_M))[:, None] - start_q
                k_rel = (start_k + k_start + tl.arange(0, BLOCK_N))[None, :] - start_k
                is_sink = k_rel < sink_num
                is_local = k_rel >= q_rel + (seqlen_k - seqlen_q) - local_num + 1
                is_valid_qk = (is_sink | is_local) & (q_rel + (seqlen_k - seqlen_q) >= k_rel) & (k_rel < seqlen_k)
                qk = tl.where(is_valid_qk, qk, -1e38)
            else:
                is_valid_qk = ((m_idx * BLOCK_M + tl.arange(0, BLOCK_M))[:, None] >= (k_start + tl.arange(0, BLOCK_N))[None, :]) & ((start_k + k_start + tl.arange(0, BLOCK_N))[None, :] < end_k)
                qk = tl.where(is_valid_qk, qk, -1e38)
            
            m_new = tl.max(qk, axis=1)
            m_next = tl.maximum(m_i, m_new)
            alpha = tl.math.exp(m_i - m_next)
            beta = tl.math.exp(qk - m_next[:, None])
            beta = tl.where(is_valid_qk, beta, 0.0)
            
            o_i = o_i * alpha[:, None]
            o_i = tl.dot(beta.to(v_chunk.dtype), v_chunk, o_i)
            
            d_new = tl.sum(beta, axis=1)
            d_i = d_i * alpha + d_new
            m_i = m_next
            
        # Local blocks
        if EXACT_STREAMING:
            local_start_blk = tl.maximum(sink_blocks_cdiv, (m_idx * BLOCK_M + seqlen_k - seqlen_q - local_num + 1) // BLOCK_N)
            local_end_blk = tl.cdiv((m_idx + 1) * BLOCK_M + seqlen_k - seqlen_q, BLOCK_N)
        else:
            local_start_blk = tl.maximum(sink_blocks_cdiv, m_idx - local_blocks_cdiv + 1)
            local_end_blk = m_idx + 1
            
        max_k_blocks = tl.cdiv(seqlen_k, BLOCK_N)
        local_end_blk_bounded = tl.minimum(local_end_blk, max_k_blocks)
        local_start_blk_bounded = tl.minimum(local_start_blk, local_end_blk_bounded)
        for k_blk in range(local_start_blk_bounded, local_end_blk_bounded):
            k_start = k_blk * BLOCK_N
            k_offset = (start_k + k_start + tl.arange(0, BLOCK_N))[None, :] * stride_k_tok + kv_head_idx * stride_k_head + tl.arange(0, BLOCK_DMODEL)[:, None] * stride_k_dim
            v_offset = (start_k + k_start + tl.arange(0, BLOCK_N))[:, None] * stride_v_tok + kv_head_idx * stride_v_head + tl.arange(0, BLOCK_DMODEL)[None, :] * stride_v_dim
            k_chunk = tl.load(K + k_offset, mask=(start_k + k_start + tl.arange(0, BLOCK_N))[None, :] < end_k, other=0.0)
            v_chunk = tl.load(V + v_offset, mask=(start_k + k_start + tl.arange(0, BLOCK_N))[:, None] < end_k, other=0.0)
            
            qk = tl.dot(q_chunk, k_chunk) * softmax_scale
            if EXACT_STREAMING:
                q_rel = (start_q + m_idx * BLOCK_M + tl.arange(0, BLOCK_M))[:, None] - start_q
                k_rel = (start_k + k_start + tl.arange(0, BLOCK_N))[None, :] - start_k
                is_sink = k_rel < sink_num
                is_local = k_rel >= q_rel + (seqlen_k - seqlen_q) - local_num + 1
                is_valid_qk = (is_sink | is_local) & (q_rel + (seqlen_k - seqlen_q) >= k_rel) & (k_rel < seqlen_k)
                qk = tl.where(is_valid_qk, qk, -1e38)
            else:
                is_valid_qk = ((m_idx * BLOCK_M + tl.arange(0, BLOCK_M))[:, None] >= (k_start + tl.arange(0, BLOCK_N))[None, :]) & ((start_k + k_start + tl.arange(0, BLOCK_N))[None, :] < end_k)
                qk = tl.where(is_valid_qk, qk, -1e38)
            
            m_new = tl.max(qk, axis=1)
            m_next = tl.maximum(m_i, m_new)
            alpha = tl.math.exp(m_i - m_next)
            beta = tl.math.exp(qk - m_next[:, None])
            beta = tl.where(is_valid_qk, beta, 0.0)
            
            o_i = o_i * alpha[:, None]
            o_i = tl.dot(beta.to(v_chunk.dtype), v_chunk, o_i)
            
            d_new = tl.sum(beta, axis=1)
            d_i = d_i * alpha + d_new
            m_i = m_next
            
    # ------------------ Blocksparse ------------------
    elif m_type == 3:
        bs_head_idx = tl.load(head_mask_type + head_idx) - 1
        mask_row_ptr = blockmask + (batch_idx * num_blocksparse_heads * q_slices + bs_head_idx * q_slices + m_idx) * k_slices
        for k_slice_idx in range(0, k_slices):
            k_blk = tl.load(mask_row_ptr + k_slice_idx)
            is_valid_blk = k_blk != -1
            
            k_start = tl.where(is_valid_blk, k_blk * BLOCK_N, 0)
            k_offset = (start_k + k_start + tl.arange(0, BLOCK_N))[None, :] * stride_k_tok + kv_head_idx * stride_k_head + tl.arange(0, BLOCK_DMODEL)[:, None] * stride_k_dim
            v_offset = (start_k + k_start + tl.arange(0, BLOCK_N))[:, None] * stride_v_tok + kv_head_idx * stride_v_head + tl.arange(0, BLOCK_DMODEL)[None, :] * stride_v_dim
            
            load_k_mask = ((start_k + k_start + tl.arange(0, BLOCK_N))[None, :] < end_k) & is_valid_blk
            load_v_mask = ((start_k + k_start + tl.arange(0, BLOCK_N))[:, None] < end_k) & is_valid_blk
            
            k_chunk = tl.load(K + k_offset, mask=load_k_mask, other=0.0)
            v_chunk = tl.load(V + v_offset, mask=load_v_mask, other=0.0)
            
            qk = tl.dot(q_chunk, k_chunk) * softmax_scale
            q_rel = (start_q + m_idx * BLOCK_M + tl.arange(0, BLOCK_M))[:, None] - start_q
            k_rel = (start_k + k_start + tl.arange(0, BLOCK_N))[None, :] - start_k
            is_valid_qk = (q_rel + (seqlen_k - seqlen_q) >= k_rel) & (k_rel < seqlen_k) & is_valid_blk
            qk = tl.where(is_valid_qk, qk, -1e38)
            
            m_new = tl.max(qk, axis=1)
            m_next = tl.maximum(m_i, m_new)
            alpha = tl.math.exp(m_i - m_next)
            beta = tl.math.exp(qk - m_next[:, None])
            
            o_i = o_i * alpha[:, None]
            o_i = tl.dot(beta.to(v_chunk.dtype), v_chunk, o_i)
            
            d_new = tl.sum(beta, axis=1)
            d_i = d_i * alpha + d_new
            m_i = m_next
            
    d_i_safe = tl.where(d_i > 0.0, d_i, 1.0)
    o_final = o_i / d_i_safe[:, None]
    o_offset = (start_q + m_idx * BLOCK_M + tl.arange(0, BLOCK_M))[:, None] * stride_o_tok + head_idx * stride_o_head + tl.arange(0, BLOCK_DMODEL)[None, :] * stride_o_dim
    tl.store(Out + o_offset, o_final.to(Out.dtype.element_ty), mask=q_mask)
    
    lse_val = m_i + tl.math.log(d_i)
    lse_offset = (batch_idx * num_heads + head_idx) * max_seqlen_q + m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    tl.store(LSE + lse_offset, lse_val, mask=(m_idx * BLOCK_M + tl.arange(0, BLOCK_M)) < seqlen_q)


def triton_block_sparse_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    m_block_dim: int,
    n_block_dim: int,
    head_mask_type: torch.Tensor,
    streaming_info: torch.Tensor,
    row_blockmask: Optional[torch.Tensor],
    max_seqlen_q_: int,
    max_seqlen_k_: int,
    p_dropout: float,
    softmax_scale: float,
    is_causal: bool,
    exact_streaming: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    
    total_q, num_heads, head_dim = q.shape
    total_k, num_heads_kv, _ = k.shape
    batch_size = cu_seqlens_q.shape[0] - 1
    
    max_seqlen_q_rounded = (max_seqlen_q_ + 127) // 128 * 128
    
    out = torch.zeros_like(q)
    softmax_lse = torch.empty((batch_size, num_heads, max_seqlen_q_rounded), dtype=torch.float32, device=q.device)
    
    q_slices = (max_seqlen_q_rounded + m_block_dim - 1) // m_block_dim
    k_slices = (max_seqlen_k_ + n_block_dim - 1) // n_block_dim
    num_blocksparse_heads = row_blockmask.shape[1] if row_blockmask is not None else 0
    
    grid = (batch_size, num_heads, triton.cdiv(max_seqlen_q_rounded, 128))
    
    _block_sparse_attn_fwd_kernel[grid](
        q, k, v, out, softmax_lse,
        cu_seqlens_q, cu_seqlens_k,
        head_mask_type,
        streaming_info,
        row_blockmask if row_blockmask is not None else q,
        softmax_scale,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        max_seqlen_q_rounded, max_seqlen_k_,
        num_heads, num_blocksparse_heads, q_slices, k_slices,
        BLOCK_M=128,
        BLOCK_N=128,
        BLOCK_DMODEL=head_dim,
        HQ_OVER_HKV=num_heads // num_heads_kv,
        EXACT_STREAMING=exact_streaming,
        num_warps=4,
        num_stages=2,
    )
    
    return out, softmax_lse
