// Auto-generated at 2026-10-29 00:14:09.780582 by ops-translator legacy

__constant__ int dims_initialise_chunk_kernel_x [3][1];
static int dims_initialise_chunk_kernel_x_h [3][1] = {{0}};

//  =============
//  User function
//  =============
__device__ void initialise_chunk_kernel_x_gpu(ACC<double> &vertexx,
  const ACC<int> &xx,
  ACC<double> &vertexdx) {

  int x_min=field.x_min-2;
  double min_x, d_x;

  d_x = (grid.xmax - grid.xmin)/(double)grid.x_cells;
  min_x=grid.xmin+d_x*field.left;

  vertexx(0,0) = min_x + d_x * (xx(0,0) - x_min);
  vertexdx(0,0) = (double)d_x;
}


//  ============================
//  Cuda kernel wrapper function
//  ============================
__global__ void ops_initialise_chunk_kernel_x(
double* __restrict arg0,
int* __restrict arg1,
double* __restrict arg2,
int size0,
int size1) {
    int idx_y = blockDim.y * blockIdx.y + threadIdx.y;
    int idx_x = blockDim.x * blockIdx.x + threadIdx.x;

    arg0 += idx_x * 1*1 + idx_y * 0*1 * dims_initialise_chunk_kernel_x[0][0];
    arg1 += idx_x * 1*1 + idx_y * 0*1 * dims_initialise_chunk_kernel_x[1][0];
    arg2 += idx_x * 1*1 + idx_y * 0*1 * dims_initialise_chunk_kernel_x[2][0];

    if (idx_x < size0 && idx_y < size1) {
        ACC<double> argp0(dims_initialise_chunk_kernel_x[0][0], arg0);
        const ACC<int> argp1(dims_initialise_chunk_kernel_x[1][0], arg1);
        ACC<double> argp2(dims_initialise_chunk_kernel_x[2][0], arg2);
        initialise_chunk_kernel_x_gpu(
         argp0, argp1, argp2);
    }

} //End of cuda kernel wrapper function

//  ==================
//  Host stub function
//  ==================
#ifndef OPS_LAZY
void ops_par_loop_initialise_chunk_kernel_x(char const *name, ops_block block, int dim, int* range,
 ops_arg arg0, ops_arg arg1, ops_arg arg2) {
#else
void ops_par_loop_initialise_chunk_kernel_x_execute(ops_kernel_descriptor *desc) {
    ops_block block = desc->block;
    int dim = desc->dim;
    int *range = desc->range;
    ops_arg arg0 = desc->args[0];
    ops_arg arg1 = desc->args[1];
    ops_arg arg2 = desc->args[2];
#endif

//  ======
//  Timing
//  ======
    double __t1, __t2, __c1, __c2;

    ops_arg args[3];

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;

#if CHECKPOINTING && !OPS_LAZY
    if (!ops_checkpointing_before(args, 3, range, 7)) return;
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timing_realloc(block->instance, 7, "initialise_chunk_kernel_x");
        block->instance->OPS_kernels[7].count++;
        ops_timers_core(&__c1, &__t1);
    }

//  =================================================
//  compute locally allocated range for the sub-block
//  =================================================
    int start_indx[2];
    int end_indx[2];
#ifdef OPS_MPI
    int arg_idx[2];
#endif

#if defined(OPS_LAZY) || !defined(OPS_MPI)
    for (int n = 0; n < 2; n++) {
        start_indx[n] = range[2*n];
        end_indx[n]   = range[2*n+1];
    }
#else
    if (compute_ranges(args, 3, block, range, start_indx, end_indx, arg_idx) < 0) return;
#endif

    int xdim0 = args[0].dat->size[0];
    int xdim1 = args[1].dat->size[0];
    int xdim2 = args[2].dat->size[0];

    if (xdim0 != dims_initialise_chunk_kernel_x_h[0][0] || xdim1 != dims_initialise_chunk_kernel_x_h[1][0] || xdim2 != dims_initialise_chunk_kernel_x_h[2][0]) {
        dims_initialise_chunk_kernel_x_h[0][0] = xdim0;
        dims_initialise_chunk_kernel_x_h[1][0] = xdim1;
        dims_initialise_chunk_kernel_x_h[2][0] = xdim2;

        hipSafeCall(block->instance->ostream(), hipMemcpyToSymbol( dims_initialise_chunk_kernel_x, dims_initialise_chunk_kernel_x_h, sizeof(dims_initialise_chunk_kernel_x)));
    }


    int x_size = MAX(0,end_indx[0]-start_indx[0]);
    int y_size = MAX(0,end_indx[1]-start_indx[1]);

    dim3 grid( (x_size-1)/block->instance->OPS_block_size_x+ 1, (y_size-1)/block->instance->OPS_block_size_y + 1, 1);
    dim3 tblock(block->instance->OPS_block_size_x,block->instance->OPS_block_size_y,block->instance->OPS_block_size_z);



    long long int dat0 = (block->instance->OPS_soa ? args[0].dat->type_size : args[0].dat->elem_size);
    long long int dat1 = (block->instance->OPS_soa ? args[1].dat->type_size : args[1].dat->elem_size);
    long long int dat2 = (block->instance->OPS_soa ? args[2].dat->type_size : args[2].dat->elem_size);

    char *p_a[3];

//  =======================
//  set up initial pointers
//  =======================
    long long int base0 = args[0].dat->base_offset + 
             dat0 * 1 * (start_indx[0] * args[0].stencil->stride[0]);
    base0 = base0 + dat0 *
                       args[0].dat->size[0] *
                      (start_indx[1] * args[0].stencil->stride[1]);
    p_a[0] = (char *)args[0].data_d + base0;

    long long int base1 = args[1].dat->base_offset + 
             dat1 * 1 * (start_indx[0] * args[1].stencil->stride[0]);
    base1 = base1 + dat1 *
                       args[1].dat->size[0] *
                      (start_indx[1] * args[1].stencil->stride[1]);
    p_a[1] = (char *)args[1].data_d + base1;

    long long int base2 = args[2].dat->base_offset + 
             dat2 * 1 * (start_indx[0] * args[2].stencil->stride[0]);
    base2 = base2 + dat2 *
                       args[2].dat->size[0] *
                      (start_indx[1] * args[2].stencil->stride[1]);
    p_a[2] = (char *)args[2].data_d + base2;

//  =============
//  Halo exchange
//  =============
#ifndef OPS_LAZY
    ops_H_D_exchanges_device(args, 3);
    ops_halo_exchanges(args, 3, range);
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[7].mpi_time += __t2 - __t1;
    }


//call kernel wrapper function, passing in pointers to data
    if (x_size > 0 && y_size > 0) {

        ops_initialise_chunk_kernel_x<<<grid, tblock >>> ( 
         (double *)p_a[0], (int *)p_a[1],
         (double *)p_a[2],
        x_size, y_size);

    }

    hipSafeCall(block->instance->ostream(), hipGetLastError());

    if (block->instance->OPS_diags>1) {
        hipSafeCall(block->instance->ostream(), hipDeviceSynchronize());
        ops_timers_core(&__c1, &__t1);
        block->instance->OPS_kernels[7].time += __t1 - __t2;
    }

#ifndef OPS_LAZY
    ops_set_dirtybit_device(args, 3);
    ops_set_halo_dirtybit3(&args[0], range);
    ops_set_halo_dirtybit3(&args[2], range);
#endif

    if (block->instance->OPS_diags > 1) {
        //  ====================
        //  Update kernel record
        //  ====================
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[7].mpi_time += __t2 - __t1;
        block->instance->OPS_kernels[7].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg0);
        block->instance->OPS_kernels[7].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg1);
        block->instance->OPS_kernels[7].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg2);
    }
}

#ifdef OPS_LAZY
void ops_par_loop_initialise_chunk_kernel_x(char const *name, ops_block block, int dim, int* range,
 ops_arg arg0, ops_arg arg1, ops_arg arg2)
{
    ops_arg args[3];

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;

    create_kerneldesc_and_enque("initialise_chunk_kernel_x", args, 3, 7, dim, 1, range, block, ops_par_loop_initialise_chunk_kernel_x_execute);
}
#endif
