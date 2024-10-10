# Adapted from https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_blocksparse_attn_interface.py

import block_sparse_attn_cuda
import torch
import torch.nn as nn


def convert_blockmask(blockmask, causal):
    """Convert from the 0-1 format to the format used by the CUDA code.
    0 means the block is skipped.
    nonzero means the block is not skipped.
    Argument:
        blockmask: (row, col): a 0-1 tensor
    Return:
        blockmask_converted: (col, row), dtype torch.int32: for each column, it contains the row
            indices of the nonzero blocks, padded with -1 to reach length @row.
            The indices are multiplied by 4, with the smallest bit used to encode whether
            it is the first nonzero in its row, and the 2nd smallest bit to encode whether it is
            the last nonzero in its row..
    """
    assert not causal
    nrow, ncol = blockmask.shape
    # Sort does not support bool on CUDA
    blockmask = blockmask.to(dtype=torch.uint8)
    nonzero_val, nonzero_sorted_rowidx = blockmask.sort(dim=0, stable=True, descending=True)
    nonzero_unsorted_rowidx = nonzero_sorted_rowidx.argsort(dim=0)
    last_nonzero_col_per_row = blockmask.sort(dim=-1, stable=True).indices[:, -1]
    last_nonzero_col_per_row_after_sort = nonzero_unsorted_rowidx[
        torch.arange(nrow, device=blockmask.device), last_nonzero_col_per_row
    ]
    first_nonzero_col_per_row = blockmask.sort(dim=-1, stable=True, descending=True).indices[:, 0]
    first_nonzero_col_per_row_after_sort = nonzero_unsorted_rowidx[
        torch.arange(nrow, device=blockmask.device), first_nonzero_col_per_row
    ]
    nonzero_idx = nonzero_sorted_rowidx * 4
    nonzero_idx[last_nonzero_col_per_row_after_sort, last_nonzero_col_per_row] += 2
    nonzero_idx[first_nonzero_col_per_row_after_sort, first_nonzero_col_per_row] += 1
    nonzero_idx[nonzero_val == 0] = -1
    return nonzero_idx.T.contiguous().to(dtype=torch.int32)


def convert_blockmask_row_reverse(blockmask, causal=False):
    # assert not causal
    # nrow, ncol = blockmask.shape
    # Sort does not support bool on CUDA
    blockmask = blockmask.to(dtype=torch.uint8)
    nonzero_val, nonzero_sorted_rowidx = blockmask.sort(dim=-1, stable=True, descending=False)
    
    nonzero_idx = nonzero_sorted_rowidx
    nonzero_idx[nonzero_val == 0] = -1
    # print("nonzero_idx: ", nonzero_idx)
    nonzero_idx = torch.flip(nonzero_idx, dims=[-1])
    # print("nonzero_idx: ", nonzero_idx)
    
    return nonzero_idx.contiguous().to(dtype=torch.int32)


def convert_blockmask_col_reverse(blockmask, causal=False):
    # assert not causal
    # nrow, ncol = blockmask.shape
    # Sort does not support bool on CUDA
    blockmask = blockmask.to(dtype=torch.uint8)
    nonzero_val, nonzero_sorted_rowidx = blockmask.sort(dim=-2, stable=True, descending=False)
    
    nonzero_idx = nonzero_sorted_rowidx
    nonzero_idx[nonzero_val == 0] = -1
    nonzero_idx = torch.flip(nonzero_idx, dims=[-2])
    nonzero_idx = torch.transpose(nonzero_idx, -1, -2)
    
    return nonzero_idx.contiguous().to(dtype=torch.int32)


def replace_ones_with_count(tensor):
    ones_mask = tensor == 1
    ones_num = ones_mask.sum()
    count = torch.cumsum(ones_mask, dim=-1).to(tensor.dtype)
    count = count * ones_mask
    tensor = tensor.masked_scatter(ones_mask, count[ones_mask])
    return tensor, ones_num


def _block_sparse_attn_forward(
    q, k, v,
    cu_seqlens_q, cu_seqlens_k,
    m_block_dim, n_block_dim,
    head_mask_type,
    streaming_info,
    row_blockmask,
    max_seqlen_q_, max_seqlen_k_,
    p_dropout,
    softmax_scale,
    is_causal,
    exact_streaming,
    return_softmax,
    window_size_left,
    window_size_right
):
    out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = block_sparse_attn_cuda.fwd_block(
        q, k, v,
        cu_seqlens_q, cu_seqlens_k,
        m_block_dim, n_block_dim,
        head_mask_type,
        streaming_info,
        row_blockmask,
        max_seqlen_q_, max_seqlen_k_,
        p_dropout,
        softmax_scale,
        is_causal,
        exact_streaming,
        return_softmax,
        window_size_left,
        window_size_right, 
        None
    )
    return out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state


def _block_sparse_attn_backward(
    dout,
    q, k, v,
    out,
    softmax_lse,
    dq, dk, dv,
    cu_seqlens_q, cu_seqlens_k,
    m_block_dim, n_block_dim,
    head_mask_type,
    streaming_info,
    col_blockmask,
    max_seqlen_q_, max_seqlen_k_,
    p_dropout,
    softmax_scale,
    zero_tensors,
    is_causal,
    window_size_left,
    window_size_right,
    deterministic,
    rng_state=None,
):
    dq, dk, dv, softmax_d = block_sparse_attn_cuda.bwd_block(
        dout,
        q, k, v,
        out,
        softmax_lse,
        dq, dk, dv,
        cu_seqlens_q, cu_seqlens_k,
        m_block_dim, n_block_dim,
        head_mask_type,
        streaming_info,
        col_blockmask,
        max_seqlen_q_, max_seqlen_k_,
        p_dropout,
        softmax_scale,
        zero_tensors,
        is_causal,
        window_size_left,
        window_size_right,
        deterministic,
        None, rng_state
    )
    return dq, dk, dv, softmax_d


class BlockSparseAttnFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                q, k, v,
                cu_seqlens_q, cu_seqlens_k,
                m_block_dim, n_block_dim,
                head_mask_type,
                streaming_info,
                base_blockmask,
                max_seqlen_q_, max_seqlen_k_,
                p_dropout,
                softmax_scale,
                is_causal,
                exact_streaming,
                return_softmax,
                window_size_left,
                window_size_right, deterministic=False):
        # Save rng_state because the backward pass will regenerate the dropout mask
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        if base_blockmask is not None:
            row_blockmask = convert_blockmask_row_reverse(base_blockmask, is_causal)
        else:
            row_blockmask = None
        
        if exact_streaming:
            assert streaming_info is not None
            assert is_causal
        
        out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = _block_sparse_attn_forward(
            q, k, v,
            cu_seqlens_q, cu_seqlens_k,
            m_block_dim, n_block_dim,
            head_mask_type,
            streaming_info,
            row_blockmask,
            max_seqlen_q_, max_seqlen_k_,
            p_dropout,
            softmax_scale,
            is_causal,
            exact_streaming,
            return_softmax=False,
            window_size_left=window_size_left,
            window_size_right=window_size_right
        )
        ctx.save_for_backward(q, k, v,
                              out, S_dmask, softmax_lse,
                              cu_seqlens_q, cu_seqlens_k,
                              head_mask_type,
                              streaming_info,
                              base_blockmask,
                              rng_state)
        # ctx.is_blocksparse = is_blocksparse
        ctx.m_block_dim = m_block_dim
        ctx.n_block_dim = n_block_dim
        ctx.window_size_left = window_size_left
        ctx.window_size_right = window_size_right
        ctx.max_seqlen_q_ = max_seqlen_q_
        ctx.max_seqlen_k_ = max_seqlen_k_
        ctx.p_dropout = p_dropout
        ctx.softmax_scale = softmax_scale
        ctx.is_causal = is_causal
        ctx.exact_streaming = exact_streaming
        ctx.deterministic = deterministic
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, S_dmask, softmax_lse, cu_seqlens_q, cu_seqlens_k, head_mask_type, streaming_info, base_blockmask, rng_state = ctx.saved_tensors
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        # S_dmask is None, temporarily use another tensor just to get it running
        if base_blockmask is not None:
            col_blockmask = convert_blockmask_col_reverse(base_blockmask, ctx.is_causal)
        else:
            col_blockmask = None
            
        assert not ctx.exact_streaming, "Exact streaming not supported in backward pass"
            
        _block_sparse_attn_backward(
            dout,
            q, k, v,
            out,
            softmax_lse,
            dq, dk, dv,
            cu_seqlens_q, cu_seqlens_k,
            ctx.m_block_dim, ctx.n_block_dim,
            head_mask_type,
            streaming_info,
            col_blockmask,
            ctx.max_seqlen_q_, ctx.max_seqlen_k_,
            ctx.p_dropout,
            ctx.softmax_scale,
            True,  # zero_tensors
            ctx.is_causal,
            ctx.window_size_left,
            ctx.window_size_right,
            ctx.deterministic,
            rng_state=rng_state
        )
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None


# We duplicate code to return both the output and the softmax for testing
# Returning both makes backward a bit slower, so we want to keep using the other version for speed.
class BlockSparseAttnFunWithS(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                q, k, v,
                cu_seqlens_q, cu_seqlens_k,
                m_block_dim, n_block_dim,
                head_mask_type,
                streaming_info,
                base_blockmask,
                max_seqlen_q_, max_seqlen_k_,
                p_dropout,
                softmax_scale,
                is_causal,
                exact_streaming,
                return_softmax,
                window_size_left,
                window_size_right,
                deterministic=False):
        # Save rng_state because the backward pass will regenerate the dropout mask
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        if base_blockmask is not None:
            row_blockmask = convert_blockmask_row_reverse(base_blockmask, is_causal)
        else:
            row_blockmask = None
            
        if exact_streaming:
            assert streaming_info is not None
            print("is_causal: ", is_causal)
            assert is_causal
        
        out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = _block_sparse_attn_forward(
            q, k, v,
            cu_seqlens_q, cu_seqlens_k,
            m_block_dim, n_block_dim,
            head_mask_type,
            streaming_info,
            row_blockmask,
            max_seqlen_q_, max_seqlen_k_,
            p_dropout,
            softmax_scale,
            is_causal,
            exact_streaming,
            return_softmax=return_softmax and p_dropout > 0,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
        )
        
        ctx.save_for_backward(q, k, v,
                              out, softmax_lse,
                              cu_seqlens_q, cu_seqlens_k,
                              head_mask_type,
                              streaming_info,
                              base_blockmask,
                              rng_state)
        # ctx.is_blocksparse = is_blocksparse
        ctx.m_block_dim = m_block_dim
        ctx.n_block_dim = n_block_dim
        ctx.window_size_left = window_size_left
        ctx.window_size_right = window_size_right
        ctx.max_seqlen_q_ = max_seqlen_q_
        ctx.max_seqlen_k_ = max_seqlen_k_
        ctx.p_dropout = p_dropout
        ctx.softmax_scale = softmax_scale
        ctx.is_causal = is_causal
        ctx.exact_streaming = exact_streaming
        ctx.deterministic = deterministic
        return out, softmax_lse, S_dmask

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, head_mask_type, streaming_info, base_blockmask, rng_state = ctx.saved_tensors
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        
        # S_dmask is None, temporarily use another tensor just to get it running
        if base_blockmask is not None:
            col_blockmask = convert_blockmask_col_reverse(base_blockmask, ctx.is_causal)
        else:
            col_blockmask = None
        
        assert not ctx.exact_streaming, "Exact streaming not supported in backward pass"
        
        dq, dk, dv, _ = _block_sparse_attn_backward(
            dout,
            q, k, v,
            out,
            softmax_lse,
            dq, dk, dv,
            cu_seqlens_q, cu_seqlens_k,
            ctx.m_block_dim, ctx.n_block_dim,
            head_mask_type,
            streaming_info,
            col_blockmask,
            ctx.max_seqlen_q_, ctx.max_seqlen_k_,
            ctx.p_dropout,
            ctx.softmax_scale,
            True,  # zero_tensors
            ctx.is_causal,
            ctx.window_size_left,
            ctx.window_size_right,
            ctx.deterministic,
            rng_state=rng_state
        )
        
        dq = dq[..., : dout.shape[-1]]  # We could have padded the head dimension
        dk = dk[..., : dout.shape[-1]]
        dv = dv[..., : dout.shape[-1]]
        
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None


def block_sparse_attn_func(
    q, k, v,
    cu_seqlens_q, cu_seqlens_k,
    head_mask_type,
    streaming_info,
    base_blockmask,
    max_seqlen_q_, max_seqlen_k_,
    p_dropout,
    deterministic=False,
    softmax_scale=None,
    is_causal=False,
    exact_streaming=False,
    return_attn_probs=False,
):
    head_mask_type, blocksparse_head_num = replace_ones_with_count(head_mask_type)
    if base_blockmask is not None:
        assert base_blockmask.shape[1] == blocksparse_head_num
    
    """dropout_p should be set to 0.0 during evaluation"""
    # print("is_causal0: ", is_causal)
    func = BlockSparseAttnFun if not return_attn_probs else BlockSparseAttnFunWithS
    return func.apply(
                q, k, v,
                cu_seqlens_q, cu_seqlens_k,
                128, 128,
                head_mask_type,
                streaming_info,
                base_blockmask,
                max_seqlen_q_, max_seqlen_k_,
                p_dropout,
                softmax_scale,
                is_causal,
                exact_streaming,
                return_attn_probs,
                -1, -1,
                deterministic
                )
    
    
def token_streaming_attn_func(
    q, k, v,
    cu_seqlens_q, cu_seqlens_k,
    head_mask_type,
    streaming_info,
    max_seqlen_q_, max_seqlen_k_,
    deterministic=False,
    softmax_scale=None,
    return_attn_probs=False,
):
    """dropout_p should be set to 0.0 during evaluation"""
    # print("is_causal0: ", is_causal)
    func = BlockSparseAttnFun if not return_attn_probs else BlockSparseAttnFunWithS
    return func.apply(
                q, k, v,
                cu_seqlens_q, cu_seqlens_k,
                128, 128,
                head_mask_type,
                streaming_info,
                None,
                max_seqlen_q_, max_seqlen_k_,
                0.0,
                softmax_scale,
                True,
                True,
                return_attn_probs,
                -1, -1,
                deterministic
                )
    
def block_streaming_attn_func(
    q, k, v,
    cu_seqlens_q, cu_seqlens_k,
    head_mask_type,
    streaming_info,
    max_seqlen_q_, max_seqlen_k_,
    p_dropout,
    deterministic=False,
    softmax_scale=None,
    is_causal=True,
    return_attn_probs=False,
):
    """dropout_p should be set to 0.0 during evaluation"""
    # print("is_causal0: ", is_causal)
    func = BlockSparseAttnFun if not return_attn_probs else BlockSparseAttnFunWithS
    return func.apply(
                q, k, v,
                cu_seqlens_q, cu_seqlens_k,
                128, 128,
                head_mask_type,
                streaming_info,
                None,
                max_seqlen_q_, max_seqlen_k_,
                p_dropout,
                softmax_scale,
                is_causal,
                False,
                return_attn_probs,
                -1, -1,
                deterministic
                )