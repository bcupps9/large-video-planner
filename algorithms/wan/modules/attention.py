# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import warnings

import torch

# BEGIN 3dConsistency flash-attn fallback
# Catch broad import failures (not only ModuleNotFoundError). On older HPC
# nodes flash-attn can be installed but still fail import with GLIBC errors.
try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except Exception as e:
    FLASH_ATTN_3_AVAILABLE = False
    warnings.warn(
        f"[3dConsistency fallback] flash_attn_interface unavailable; "
        f"falling back to FA2/SDPA path: {e}"
    )

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except Exception as e:
    FLASH_ATTN_2_AVAILABLE = False
    warnings.warn(
        f"[3dConsistency fallback] flash_attn unavailable; "
        f"falling back to SDPA path: {e}"
    )
# END 3dConsistency flash-attn fallback

__all__ = [
    'flash_attention',
    'attention',
]


def _sdpa_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    q_scale=None,
    causal=False,
    dtype=torch.bfloat16,
):
    # BEGIN 3dConsistency fallback path
    # Match existing behavior: variable-length masks are ignored in SDPA mode.
    if q_lens is not None or k_lens is not None:
        warnings.warn(
            '[3dConsistency fallback] Variable-length mask is disabled when '
            'using scaled_dot_product_attention.'
        )

    if q_scale is not None:
        q = q * q_scale

    q = q.transpose(1, 2).to(dtype)
    k = k.transpose(1, 2).to(dtype)
    v = v.transpose(1, 2).to(dtype)

    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=None, is_causal=causal, dropout_p=dropout_p
    )
    return out.transpose(1, 2).contiguous()
    # END 3dConsistency fallback path


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    q_in, k_in, v_in = q, k, v

    # BEGIN 3dConsistency fallback path
    # Some call sites invoke flash_attention(...) directly (not attention(...)).
    # If flash-attn is unavailable, fall back here instead of asserting.
    if not (FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE):
        return _sdpa_attention(
            q=q_in,
            k=k_in,
            v=v_in,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            q_scale=q_scale,
            causal=causal,
            dtype=dtype,
        )
    # END 3dConsistency fallback path

    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        try:
            # Note: dropout_p, window_size are not supported in FA3 now.
            x = flash_attn_interface.flash_attn_varlen_func(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                    0, dtype=torch.int32).to(q.device, non_blocking=True),
                cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                    0, dtype=torch.int32).to(q.device, non_blocking=True),
                seqused_q=None,
                seqused_k=None,
                max_seqlen_q=lq,
                max_seqlen_k=lk,
                softmax_scale=softmax_scale,
                causal=causal,
                deterministic=deterministic)[0].unflatten(0, (b, lq))
        except Exception as e:
            warnings.warn(
                f"[3dConsistency fallback] FA3 runtime failure; using SDPA: {e}"
            )
            return _sdpa_attention(
                q=q_in,
                k=k_in,
                v=v_in,
                q_lens=q_lens,
                k_lens=k_lens,
                dropout_p=dropout_p,
                causal=causal,
                dtype=dtype,
            )
    elif FLASH_ATTN_2_AVAILABLE:
        try:
            x = flash_attn.flash_attn_varlen_func(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                    0, dtype=torch.int32).to(q.device, non_blocking=True),
                cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                    0, dtype=torch.int32).to(q.device, non_blocking=True),
                max_seqlen_q=lq,
                max_seqlen_k=lk,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                deterministic=deterministic).unflatten(0, (b, lq))
        except Exception as e:
            warnings.warn(
                f"[3dConsistency fallback] FA2 runtime failure; using SDPA: {e}"
            )
            return _sdpa_attention(
                q=q_in,
                k=k_in,
                v=v_in,
                q_lens=q_lens,
                k_lens=k_lens,
                dropout_p=dropout_p,
                causal=causal,
                dtype=dtype,
            )
    else:
        return _sdpa_attention(
            q=q_in,
            k=k_in,
            v=v_in,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            causal=causal,
            dtype=dtype,
        )

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    # Keep a single entrypoint so direct and indirect call sites share fallback behavior.
    return flash_attention(
        q=q,
        k=k,
        v=v,
        q_lens=q_lens,
        k_lens=k_lens,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        q_scale=q_scale,
        causal=causal,
        window_size=window_size,
        deterministic=deterministic,
        dtype=dtype,
        version=fa_version,
    )
