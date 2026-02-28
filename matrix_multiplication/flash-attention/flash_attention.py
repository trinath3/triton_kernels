import torch 
import triton.language as tl
import triton


# Standard SDPA 

def attention(q, k, v):
    # q, k, v shape: (B, H, N, D)
    
    # 1. Transpose K for the dot product: (B, H, D, N)
    # We only want to flip the last two dimensions
    k_t = k.transpose(-2, -1) 
    
    # 2. Scaled Dot Product
    # d_k is the last dimension of q
    d_k = q.shape[-1]
    attn_weights = (q @ k_t) * (d_k ** -0.5) 
    
    # 3. Softmax along the last dimension (columns of the score matrix)
    A = torch.softmax(attn_weights, dim=-1)
    
    # 4. Multiply by V: (B, H, N, N) @ (B, H, N, D) -> (B, H, N, D)
    O = A @ v
    return O



# Define the search space
configs = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'num_warps': 8, 'num_stages': 2}),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'num_warps': 16, 'num_stages': 2}),
]

@triton.autotune(
    configs=configs,
    key=['N', 'D'], # Re-tune if sequence length or head dim changes
)
@triton.jit
def flash_attn_kernel(
    Q, K, V, Out,
    stride_qb, stride_qh, stride_qn, stride_qd,
    N, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    row_block_id = tl.program_id(2)

    q_ptr_base = Q + (batch_id * stride_qb) + (head_id * stride_qh)
    k_ptr_base = K + (batch_id * stride_qb) + (head_id * stride_qh)
    v_ptr_base = V + (batch_id * stride_qb) + (head_id * stride_qh)

    offs_m = row_block_id * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)

    q_ptrs = q_ptr_base + (offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd)
    q_block = tl.load(q_ptrs, mask=offs_m[:, None] < N, other=0.0)

    # --- Keep all accumulators in float32 ---
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    qk_scale = 1.0 / (D ** 0.5)

    offs_n = tl.arange(0, BLOCK_N)
    # K is laid out as (D, BLOCK_N) for the dot: q(M,D) @ k(D,N)
    k_ptrs = k_ptr_base + (offs_n[None, :] * stride_qn + offs_d[:, None] * stride_qd)
    v_ptrs = v_ptr_base + (offs_n[:, None] * stride_qn + offs_d[None, :] * stride_qd)

    for start_n in range(0, N, BLOCK_N):
        # Load K block: shape (D, BLOCK_N)
        k_block = tl.load(
            k_ptrs + start_n * stride_qn,
            mask=(start_n + offs_n[None, :]) < N,
            other=0.0
        )

        # q(M, D) @ k(D, N) -> qk(M, N)
        qk = tl.dot(q_block, k_block)
        qk = qk * qk_scale  # float32

        # --- Online softmax update (all float32) ---
        m_ij = tl.max(qk, axis=1)                        # (M,)
        m_i_new = tl.maximum(m_i, m_ij)                  # (M,)

        alpha = tl.exp(m_i - m_i_new)                    # (M,)  rescale factor
        p_ij = tl.exp(qk - m_i_new[:, None])             # (M, N) in float32

        l_ij = tl.sum(p_ij, axis=1)                      # (M,)
        l_i_new = alpha * l_i + l_ij                     # (M,)

        # Rescale accumulator, then add new contribution
        acc = acc * alpha[:, None]

        # Load V block: shape (BLOCK_N, D)
        v_block = tl.load(
            v_ptrs + start_n * stride_qn,
            mask=(start_n + offs_n[:, None]) < N,
            other=0.0
        )

        # Cast to fp16 ONLY for the dot (tensor cores), immediately cast result back
        acc += tl.dot(p_ij.to(tl.float16), v_block.to(tl.float16)).to(tl.float32)

        m_i = m_i_new
        l_i = l_i_new

    # Normalize
    acc = acc / l_i[:, None]

    # Write output — cast down to original dtype only at store
    out_ptrs = (
        Out
        + (batch_id * stride_qb)
        + (head_id * stride_qh)
        + (offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd)
    )
    tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < N)




def flash_attention(q, k, v):
    B, H, N, D = q.shape 
    out = torch.empty_like(q)
    
    # We still need to define the grid, but we don't know BLOCK_M yet.
    # We can use a helper or just assume a reasonable default for grid calc.
    grid = lambda META: (B, H, triton.cdiv(N, META['BLOCK_M']))

    flash_attn_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        N, D,
        # BLOCK_M and BLOCK_N are omitted here; autotune injects them
    )
    return out


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],                          # x-axis: Sequence Length
        x_vals=[128 * i for i in range(2, 33)], # Sweep from 256 to 4096
        line_arg="provider",                    
        line_vals=["torch-native", "triton"],
        line_names=["Torch (native)", "Triton"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="TFLOPS",                        # Changed to TFLOPS for better insight
        plot_name="Flash Attention Performance",
        args={"Batch": 1, "Heads": 12, "D_head": 64}, 
    )
)
def benchmark(Batch, Heads, N, D_head, provider):
    # Use the N passed from x_vals
    q = torch.randn((Batch, Heads, N, D_head), device="cuda", dtype=torch.float16)
    k = torch.randn((Batch, Heads, N, D_head), device="cuda", dtype=torch.float16)
    v = torch.randn((Batch, Heads, N, D_head), device="cuda", dtype=torch.float16)

    quantiles = [0.5, 0.2, 0.8]

    if provider == "torch-native":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: attention(q, k, v), quantiles=quantiles)
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: flash_attention(q, k, v), quantiles=quantiles)

    # Calculation for Attention TFLOPS: 
    # 2 * (Q@K) + 2 * (Softmax@V) = 4 * Batch * Heads * N^2 * D_head
    tflops = lambda ms: 4 * Batch * Heads * N**2 * D_head * 1e-12 / (ms * 1e-3)

    return tflops(ms), tflops(max_ms), tflops(min_ms)

# Run it
benchmark.run(show_plots=True, print_data=True)


