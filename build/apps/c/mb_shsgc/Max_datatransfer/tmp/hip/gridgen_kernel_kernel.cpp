// Auto-generated at 2026-10-29 00:14:11.522426 by ops-translator legacy

__constant__ int dims_gridgen_kernel [2][1];
static int dims_gridgen_kernel_h [2][1] = {{0}};

//  =============
//  User function
//  =============
__device__ void gridgen_kernel_gpu(ACC<double> &x,
  const int *idx) {

  x(0) = xt +  idx[0] *dx;

}


//  ============================
//  Cuda kernel wrapper function
//  ============================
__global__ void ops_gridgen_kernel(
double* __restrict arg0,
int arg_idx0,
int size0) {
    int idx_x = blockDim.x * blockIdx.x + threadIdx.x;

    int arg_idx[1];
    arg_idx[0] = arg_idx0+idx_x;

    arg0 += idx_x * 1*1;

    if (idx_x < size0) {
        ACC<double> argp0(arg0);
        gridgen_kernel_gpu(
         argp0, arg_idx);
    }

} //End of cuda kernel wrapper function

//  ==================
//  Host stub function
//  ==================
#ifndef OPS_LAZY
void ops_par_loop_gridgen_kernel(char const *name, ops_block block, int dim, int* range,
 ops_arg arg0, ops_arg arg1) {
#else
void ops_par_loop_gridgen_kernel_execute(ops_kernel_descriptor *desc) {
    ops_block block = desc->block;
    int dim = desc->dim;
    int *range = desc->range;
    ops_arg arg0 = desc->args[0];
    ops_arg arg1 = desc->args[1];
#endif

//  ======
//  Timing
//  ======
    double __t1, __t2, __c1, __c2;

    ops_arg args[2];

    args[0] = arg0;
    args[1] = arg1;

#if CHECKPOINTING && !OPS_LAZY
    if (!ops_checkpointing_before(args, 2, range, 0)) return;
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timing_realloc(block->instance, 0, "gridgen_kernel");
        block->instance->OPS_kernels[0].count++;
        ops_timers_core(&__c1, &__t1);
    }

//  =================================================
//  compute locally allocated range for the sub-block
//  =================================================
    int start_indx[1];
    int end_indx[1];
    int arg_idx[1];

#if defined(OPS_LAZY) || !defined(OPS_MPI)
    for (int n = 0; n < 1; n++) {
        start_indx[n] = range[2*n];
        end_indx[n]   = range[2*n+1];
    }
#else
    if (compute_ranges(args, 2, block, range, start_indx, end_indx, arg_idx) < 0) return;
#endif

#if defined(OPS_MPI)
#if defined(OPS_LAZY)
    sub_block_list sb = OPS_sub_block_list[block->index];
    arg_idx[0] = sb->decomp_disp[0]+start_indx[0];
#endif //OPS_LAZY
#else //OPS_MPI
    arg_idx[0] = start_indx[0];
#endif //OPS_MPI

    int xdim0 = args[0].dat->size[0];

    if (xdim0 != dims_gridgen_kernel_h[0][0]) {
        dims_gridgen_kernel_h[0][0] = xdim0;

        hipSafeCall(block->instance->ostream(), hipMemcpyToSymbol( dims_gridgen_kernel, dims_gridgen_kernel_h, sizeof(dims_gridgen_kernel)));
    }


    int x_size = MAX(0,end_indx[0]-start_indx[0]);

    dim3 grid( (x_size-1)/block->instance->OPS_block_size_x+ 1, 1, 1);
    dim3 tblock(block->instance->OPS_block_size_x,1,1);



    long long int dat0 = (block->instance->OPS_soa ? args[0].dat->type_size : args[0].dat->elem_size);

    char *p_a[2];

//  =======================
//  set up initial pointers
//  =======================
    long long int base0 = args[0].dat->base_offset + 
             dat0 * 1 * (start_indx[0] * args[0].stencil->stride[0]);
    p_a[0] = (char *)args[0].data_d + base0;

//  =============
//  Halo exchange
//  =============
#ifndef OPS_LAZY
    ops_H_D_exchanges_device(args, 2);
    ops_halo_exchanges(args, 2, range);
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[0].mpi_time += __t2 - __t1;
    }


//call kernel wrapper function, passing in pointers to data
    if (x_size > 0) {

        ops_gridgen_kernel<<<grid, tblock >>> ( 
         (double *)p_a[0], arg_idx[0],
        x_size);

    }

    hipSafeCall(block->instance->ostream(), hipGetLastError());

    if (block->instance->OPS_diags>1) {
        hipSafeCall(block->instance->ostream(), hipDeviceSynchronize());
        ops_timers_core(&__c1, &__t1);
        block->instance->OPS_kernels[0].time += __t1 - __t2;
    }

#ifndef OPS_LAZY
    ops_set_dirtybit_device(args, 2);
    ops_set_halo_dirtybit3(&args[0], range);
#endif

    if (block->instance->OPS_diags > 1) {
        //  ====================
        //  Update kernel record
        //  ====================
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[0].mpi_time += __t2 - __t1;
        block->instance->OPS_kernels[0].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg0);
    }
}

#ifdef OPS_LAZY
void ops_par_loop_gridgen_kernel(char const *name, ops_block block, int dim, int* range,
 ops_arg arg0, ops_arg arg1)
{
    ops_arg args[2];

    args[0] = arg0;
    args[1] = arg1;

    create_kerneldesc_and_enque("gridgen_kernel", args, 2, 0, dim, 1, range, block, ops_par_loop_gridgen_kernel_execute);
}
#endif
