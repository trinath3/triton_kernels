import torch 
import triton
import triton.language as tl 


@triton.jit
def _add_vector(
    out_ptr,
    stride_output_ptr,
    x_ptr,
    stride_x,
    y_ptr,
    num_cols,
    num_rows,
    block_size : tl.constexpr
):

    

    row_index = tl.program_id(0)

    x_start_ptr = x_ptr + (row_index * stride_x)
    y_start_ptr = y_ptr + (row_index * stride_x)
    col_offsets = tl.arange(0,block_size)

    row_mask = col_offsets < num_cols

    x_pointers = x_start_ptr + col_offsets 
    y_pointers = y_start_ptr + col_offsets



    x = tl.load(x_pointers, mask = row_mask, other = float("-inf"))
    y = tl.load(y_pointers,mask = row_mask, other = float("-inf"))

    # compute vector addition
    z = x + y 

    # write it to out_ptr
    output_row_ptr = out_ptr + (row_index * stride_output_ptr)
    output_pointers = output_row_ptr + col_offsets
    tl.store(output_pointers,z,mask = row_mask) 




def vector_add_simple(x:torch.Tensor,y:torch.Tensor) -> torch.Tensor:

    rows,cols = x.shape
    block_size = triton.next_power_of_2(cols)

    num_warps = 4 # *32 
    if block_size > 2047: # 2048 
        num_warps = 8

    if block_size > 4095: # 4096 
        num_warps = 16


    grid = (rows,)

    


    out_vector = torch.empty_like(x)

    _add_vector[grid](
        out_vector,
        out_vector.stride(0),
        x,
        x.stride(0),
        y,
        num_cols = cols,
        num_rows = rows,
        block_size = block_size,
        num_warps = num_warps
    )
    return out_vector



x = torch.rand(1,10,device = "cuda",dtype = torch.float32)
y = torch.rand(1,10,device = "cuda",dtype = torch.float32)

z = vector_add_simple(x,y)




@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],                          # x-axis: number of columns
        x_vals=[128 * i for i in range(2, 100)],  # values to sweep
        line_arg="provider",                    # the argument that selects the line
        line_vals=["torch-native", "triton"],
        line_names=["Torch (native)", "Triton"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="GB/s",
        plot_name="simple-vector-addition-performance",
        args={"M": 4096},                       # fixed arguments passed to benchmark()
    )
)
def benchmark(M,N,provider):
    x = torch.rand(M,N,device = "cuda",dtype = torch.float32)
    y = torch.rand(M,N,device = "cuda",dtype = torch.float32)
    quantiles = [0.5,0.2,0.8]

    if provider == "torch-native":
        ms,min_ms,max_ms = triton.testing.do_bench(lambda: torch.add(x,y),quantiles = quantiles)

    if provider == "triton":
        ms,min_ms,max_ms = triton.testing.do_bench(lambda: vector_add_simple(x,y),quantiles = quantiles)


    gbps = lambda ms: 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)

    return gbps(ms),gbps(max_ms), gbps(min_ms)


from pathlib import Path 
benchmark.run(show_plots = True,print_data = True, save_path = Path.cwd())
    




