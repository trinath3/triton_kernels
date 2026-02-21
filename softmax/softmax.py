import torch
import torch.nn.functional as F
import triton 
import triton.language as tl
import time 


def naive_softmax(x:torch.Tensor) -> torch.Tensor:
    """  eager mode"""
    x_max = x.max(dim = 1)[0]
    safe_x = x - x_max[:,None]
    numerator = torch.exp(safe_x)
    denominator = numerator.sum(dim=1) 
    sm_out = numerator/denominator[:,None]
    return sm_out


sample = torch.tensor([[1,2,3,4,5],[5,4,3,2,1]],dtype = torch.float32, device ="cuda") 


ref_out = torch.softmax(sample,dim =1)


print(f"{ref_out=}")




eager_out = naive_softmax(sample) 

print(f"{eager_out=}")



@triton.jit 
def _softmax_fwd_kernels(
    output_ptr,
    stride_output_row,
    input_ptr, 
    stride_input_row,
    num_cols,
    block_size:tl.constexpr,
):
    # setup input ptrs
    row_index = tl.program_id(0)

    row_start_ptr = input_ptr + (row_index * stride_input_row)
    col_offsets = tl.arange(0,block_size)
    input_pointers = row_start_ptr + col_offsets 

    row_mask = col_offsets < num_cols

    # move to SRAM 
    row = tl.load(input_pointers,mask = row_mask,other = float("-inf"))
    
    # softmax_itself
    safe_row = row - tl.max(row,axis = 0)
    numerator = tl.exp(safe_row)
    denominator = tl.sum(numerator,axis = 0)
    sm_out = numerator / denominator

    # write back to HBM 
    output_row_ptr = output_ptr + (row_index * stride_output_row)
    output_pointers = output_row_ptr + col_offsets
    tl.store(output_pointers,sm_out,mask = row_mask) 




def softmax(x:torch.Tensor) -> torch.Tensor:
    """ Triton imp of softmax, fwd pass only """ 
    rows,cols = x.shape 
    assert x.dim() == 2, f"only accepts 2D tensors for now"
    block_size = triton.next_power_of_2(cols)
    num_warps = 4 # *32 
    if block_size > 2047: # 2048 
        num_warps = 8

    if block_size > 4095: # 4096 
        num_warps = 16

    grid = (rows,)

    # allocate our output buffer 
    sm_out = torch.empty_like(x)

    _softmax_fwd_kernels[grid](
        sm_out,
        sm_out.stride(0),
        x,
        x.stride(0),
        cols,
        block_size = block_size,
        num_warps = num_warps
    )
    return sm_out




triton_out = softmax(sample)

print(f"{triton_out=}")


def online_softmax(x: torch.Tensor) -> torch.Tensor:
    """ 
    Online softmax , 2.5x faster than  eager algo
    """
    row_count,col_count = x.shape

    assert x.dim() == 2, " only 2d inputs atm"

    output = torch.zeros_like(x)

    for r in range(row_count):
        row_mask = 0 # m
        normalizer = 0 # l
        for c in range(col_count):
            curr = x[r,c]






@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],                          # x-axis: number of columns
        x_vals=[128 * i for i in range(2, 100)],  # values to sweep
        line_arg="provider",                    # the argument that selects the line
        line_vals=["torch-native", "triton", "eager"],
        line_names=["Torch (native)", "Triton", "Eager (naive)"],
        styles=[("blue", "-"), ("green", "-"), ("red", "-")],
        ylabel="GB/s",
        plot_name="softmax-performance",
        args={"M": 4096},                       # fixed arguments passed to benchmark()
    )
)
def benchmark(M,N,provider):
    x = torch.rand(M,N,device = "cuda",dtype = torch.float32)
    quantiles = [0.5,0.2,0.8]

    if provider == "torch-native":
        ms,min_ms,max_ms = triton.testing.do_bench(lambda: torch.softmax(x,axis = -1),quantiles = quantiles)

    if provider == "triton":
        ms,min_ms,max_ms = triton.testing.do_bench(lambda: softmax(x),quantiles = quantiles)


    if provider == "eager":
        ms,min_ms,max_ms = triton.testing.do_bench(lambda: naive_softmax(x),quantiles =quantiles)


    gbps = lambda ms: 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)

    return gbps(ms),gbps(max_ms), gbps(min_ms)


from pathlib import Path 
benchmark.run(show_plots = True,print_data = True, save_path = Path.cwd())
