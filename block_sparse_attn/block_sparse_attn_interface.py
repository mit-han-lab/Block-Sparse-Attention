# Adapted from https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_blocksparse_attn_interface.py

import block_sparse_attn_cuda
import torch
import torch.nn as nn
from typing import Optional, Tuple


def _get_block_size(device, head_dim, is_dropout, is_causal):
    # This should match the block sizes in the CUDA kernel
    assert head_dim <= 256
    major, minor = torch.cuda.get_device_capability(device)
    is_sm8x = major == 8 and minor > 0  # Only include sm86 and sm89, exclude sm80 (A100)
    is_sm80 = major == 8 and minor == 0
    is_sm90 = major == 9 and minor == 0
    if head_dim <= 32:
        return 128, 128
    if head_dim <= 64:
        return (128, 128) if not is_dropout else (128, 64)
    elif head_dim <= 96:
        return (64, 64) if (is_sm8x and is_causal) else (128, 64)
    elif head_dim <= 128:
        if is_sm8x:
            return (64, 64) if (not is_dropout and is_causal) else (128, 32)
        else:
            return 128, (64 if not is_dropout else 32)
    elif head_dim <= 160:
        if is_sm8x:
            return (128, 64) if not is_causal else (64, 64)
        else:
            return 128, 32
    elif head_dim <= 192:
        return (128, 64) if not is_dropout else (64, 64)
    elif head_dim <= 224:
        return (128, 64) if (is_sm80 or is_sm90) else (64, 64)
    elif head_dim <= 256:
        return (128, 64) if is_sm80 else (64, 64)
    


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


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def round_multiple(x, m):
    return (x + m - 1) // m * m


# torch.compile() support is only enabled for pytorch >= 2.4
# The reason for this is that we are using the new custom_op and register_fake
# APIs, which support inplace modification of inputs in the function itself
if torch.__version__ >= "2.4.0":
    _torch_custom_op_wrapper = torch.library.custom_op
    _torch_register_fake_wrapper = torch.library.register_fake
else:
    def noop_custom_op_wrapper(name, fn=None, /, *, mutates_args, device_types=None, schema=None):
        def wrap(func):
            return func
        if fn is None:
            return wrap
        return fn
    def noop_register_fake_wrapper(op, fn=None, /, *, lib=None, _stacklevel=1):
        def wrap(func):
            return func
        if fn is None:
            return wrap
        return fn
    _torch_custom_op_wrapper = noop_custom_op_wrapper
    _torch_register_fake_wrapper = noop_register_fake_wrapper


@_torch_custom_op_wrapper("flash_attn::_block_sparse_attn_forward", mutates_args=(), device_types="cuda")
def _block_sparse_attn_forward(
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
    return_softmax: bool,
    window_size_left: int,
    window_size_right: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    out, softmax_lse, S_dmask, rng_state = block_sparse_attn_cuda.fwd_block(
        q, k, v,
        cu_seqlens_q, cu_seqlens_k,
        head_mask_type,
        streaming_info,
        row_blockmask,
        max_seqlen_q_, max_seqlen_k_,
        p_dropout,
        softmax_scale,
        is_causal,
        window_size_left,
        window_size_right, 
        m_block_dim, n_block_dim,
        exact_streaming,
        return_softmax,
        None
    )
    return out, softmax_lse, S_dmask, rng_state


@_torch_register_fake_wrapper("flash_attn::_block_sparse_attn_forward")
def _block_sparse_attn_forward_fake(
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
    return_softmax: bool,
    window_size_left: int,
    window_size_right: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    total_q, num_heads, _ = q.shape
    out = torch.empty_like(q)
    batch_size = cu_seqlens_q.shape[0] - 1
    max_seqlen_q_rounded = round_multiple(max_seqlen_q_, 128)
    max_seqlen_k_rounded = round_multiple(max_seqlen_k_, 128)
    softmax_lse = torch.empty((batch_size, num_heads, max_seqlen_q_rounded), dtype=torch.float32, device=q.device, layout=q.layout)
    p = torch.empty((0,), dtype=q.dtype, device=q.device, layout=q.layout)
    if return_softmax:
        p = torch.empty((batch_size, num_heads, max_seqlen_q_rounded, max_seqlen_k_rounded), dtype=q.dtype, device=q.device, layout=q.layout)

    rng_state = torch.empty((2,), dtype=torch.int64, device=q.device)
    return out, softmax_lse, p, rng_state


if torch.__version__ >= "2.4.0":
    _wrapped_block_sparse_attn_forward = torch.ops.flash_attn._block_sparse_attn_forward
else:
    _wrapped_block_sparse_attn_forward = _block_sparse_attn_forward


@_torch_custom_op_wrapper("flash_attn::_block_sparse_attn_backward", mutates_args=("dq", "dk", "dv"), device_types="cuda")
def _block_sparse_attn_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    m_block_dim: int,
    n_block_dim: int,
    head_mask_type: torch.Tensor,
    streaming_info: torch.Tensor,
    col_blockmask: torch.Tensor,
    max_seqlen_q_: int,
    max_seqlen_k_: int,
    p_dropout: float,
    softmax_scale: float,
    zero_tensors: bool,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    deterministic: bool,
    rng_state: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    dout, q, k, v, out = [maybe_contiguous(x) for x in (dout, q, k, v, out)]
    dq, dk, dv, softmax_d = block_sparse_attn_cuda.bwd_block(
        dout,
        q, k, v,
        out,
        softmax_lse,
        dq, dk, dv,
        cu_seqlens_q, cu_seqlens_k,
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
        m_block_dim, n_block_dim,
        deterministic,
        None, rng_state
    )
    return softmax_d


@_torch_register_fake_wrapper("flash_attn::_block_sparse_attn_backward")
def _block_sparse_attn_backward_fake(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    m_block_dim: int,
    n_block_dim: int,
    head_mask_type: torch.Tensor,
    streaming_info: torch.Tensor,
    col_blockmask: torch.Tensor,
    max_seqlen_q_: int,
    max_seqlen_k_: int,
    p_dropout: float,
    softmax_scale: float,
    zero_tensors: bool,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    deterministic: bool,
    rng_state: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    total_q, num_heads, _ = q.shape
    batch_size = cu_seqlens_q.shape[0] - 1
    max_seqlen_q_rounded = round_multiple(max_seqlen_q_, 128)
    max_seqlen_k_rounded = round_multiple(max_seqlen_k_, 128)
    softmax_d = torch.empty((batch_size, num_heads, max_seqlen_q_rounded), device=q.device, dtype=torch.float32)
    return softmax_d


if torch.__version__ >= "2.4.0":
    _wrapped_block_sparse_attn_backward = torch.ops.flash_attn._block_sparse_attn_backward
else:
    _wrapped_block_sparse_attn_backward = _block_sparse_attn_backward


class BlockSparseAttnFunc(torch.autograd.Function):
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
                deterministic=False,
                is_grad_enabled=False):
        # Save rng_state because the backward pass will regenerate the dropout mask
        is_grad = is_grad_enabled and any(
            x.requires_grad for x in [q, k, v]
        )
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        head_size_og = q.size(2)
        if head_size_og % 8 != 0:
            q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
            k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
            v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])
        if base_blockmask is not None:
            row_blockmask = convert_blockmask_row_reverse(base_blockmask, is_causal)
        else:
            row_blockmask = None
        
        if exact_streaming:
            assert streaming_info is not None
            assert is_causal
        
        out, softmax_lse, S_dmask, rng_state = _wrapped_block_sparse_attn_forward(
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
            window_size_right=window_size_right
        )
        if is_grad:
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
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, S_dmask, softmax_lse, cu_seqlens_q, cu_seqlens_k, head_mask_type, streaming_info, base_blockmask, rng_state = ctx.saved_tensors
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        # S_dmask is None, temporarily use another tensor just to get it running
        head_size_og = dout.size(2)
        dout_padded = dout
        if head_size_og % 8 != 0:
            dout_padded = torch.nn.functional.pad(dout, [0, 8 - head_size_og % 8])
        if base_blockmask is not None:
            col_blockmask = convert_blockmask_col_reverse(base_blockmask, ctx.is_causal)
        else:
            col_blockmask = None
            
        assert not ctx.exact_streaming, "Exact streaming not supported in backward pass"
            
        _wrapped_block_sparse_attn_backward(
            dout_padded,
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
        
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None


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
    
    return BlockSparseAttnFunc.apply(
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
                deterministic,
                torch.is_grad_enabled()
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
    return BlockSparseAttnFunc.apply(
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
                deterministic,
                torch.is_grad_enabled()
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
    return BlockSparseAttnFunc.apply(
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
                deterministic,
                torch.is_grad_enabled()
                )