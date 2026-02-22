import torch 
import torch.nn 
import triton 
import triton.language as tl 



@triton.jit
def _mat_mul(out_ptr,out_stride_0,out_stride_1,
                    x_ptr,x_stride_0,x_stride_1,
                    y_ptr,y_stride_0,y_stride_1,
                    M,N,K,
                    BLOCK_SIZE_K:tl.constexpr,
                    BLOCK_SIZE_M:tl.constexpr,
                    BLOCK_SIZE_N:tl.constexpr,
                    
):
    
    
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # declare pointers
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    x_ptrs = x_ptr + (offs_m[:, None] * x_stride_0 + offs_k[None, :] * x_stride_1) # these are tile pointers
    y_ptrs = y_ptr + (offs_k[:, None] * y_stride_0 + offs_n[None, :] * y_stride_1) # these are tile pointers

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K): 
    #for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (k + offs_k[None, :] < K), other=0.0) # therse are 
        y = tl.load(y_ptrs, mask=(k + offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0) 
        acc += tl.dot(x, y,out_dtype=tl.float32) 
        # Move pointers 
        x_ptrs += BLOCK_SIZE_K * x_stride_1 
        y_ptrs += BLOCK_SIZE_K * y_stride_0 
    # Write result 
    out_ptrs = out_ptr + (offs_m[:, None] * out_stride_0 + offs_n[None, :] * out_stride_1) 
    tl.store(out_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

    


def mat_mul(x:torch.Tensor,y:torch.Tensor) -> torch.Tensor:
    assert x.shape[1] == y.shape[0], "this matrices are not compatable for mutiplication"
    M,K = x.shape
    K,N = y.shape

    x_stride_0,x_stride_1 = x.stride()
    y_stride_0,y_stride_1 = y.stride()
    out = torch.empty((M,N),device = "cuda",dtype = torch.float16)
    out_stride_0,out_stride_1 = out.stride()

    # Block sizes
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32

    # Grid definition
    grid = (
        triton.cdiv(M, BLOCK_SIZE_M),
        triton.cdiv(N, BLOCK_SIZE_N),
    )

    _mat_mul[grid](out,out_stride_0,out_stride_1,
                    x,x_stride_0,x_stride_1,
                    y,y_stride_0,y_stride_1,
                    M,N,K,
                    BLOCK_SIZE_K,BLOCK_SIZE_M,BLOCK_SIZE_N,
                    )
    
    return out




@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],                          # x-axis: number of columns
        x_vals=[128 * i for i in range(2, 100)],  # values to sweep
        line_arg="provider",                    # the argument that selects the line
        line_vals=["torch-native", "triton"],
        line_names=["Torch (native)", "Triton"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="GB/s",
        plot_name="mat_mul-performance",
        args={"M": 4096},                       # fixed arguments passed to benchmark()
    )
)
def benchmark(M,N,provider):
    x = torch.rand(M,N,device = "cuda",dtype = torch.float16)
    y = torch.rand(N,M,device = "cuda",dtype = torch.float16)
    quantiles = [0.5,0.2,0.8]

    if provider == "torch-native":
        ms,min_ms,max_ms = triton.testing.do_bench(lambda: torch.matmul(x,y),quantiles = quantiles)

    if provider == "triton":
        ms,min_ms,max_ms = triton.testing.do_bench(lambda: mat_mul(x,y),quantiles = quantiles)


    gbps = lambda ms: 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)

    return gbps(ms),gbps(max_ms), gbps(min_ms)


from pathlib import Path 
benchmark.run(show_plots = True,print_data = True, save_path = Path.cwd())
    

