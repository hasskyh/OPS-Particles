// Auto-generated at 2026-10-29 00:14:11.478458 by ops-translator legacy

__constant__ int dims_tvd_kernel [2][1];
static int dims_tvd_kernel_h [2][1] = {{0}};

//  =============
//  User function
//  =============
__device__ void tvd_kernel_gpu(const ACC<double> &tht,
  ACC<double>& ep2) {
  double maxim;
  for (int m=0; m < 3 ;m++) {
    if (tht(m,0) > tht(m,1))
      maxim = tht(m,0);
    else
      maxim = tht(m,1);
    ep2(m,0) = akap2 * maxim;
  }
}


//  ============================
//  Cuda kernel wrapper function
//  ============================
__global__ void ops_tvd_kernel(
double* __restrict arg0,
double* __restrict arg1,
int size0) {
    int idx_x = blockDim.x * blockIdx.x + threadIdx.x;

    arg0 += idx_x * 1*3;
    arg1 += idx_x * 1*3;

    if (idx_x < size0) {
        const ACC<double> argp0(3, 0, arg0);
        ACC<double> argp1(3, 0, arg1);
        tvd_kernel_gpu(
         argp0, argp1);
    }

} //End of cuda kernel wrapper function

//  ==================
//  Host stub function
//  ==================
#ifndef OPS_LAZY
void ops_par_loop_tvd_kernel(char const *name, ops_block block, int dim, int* range,
 ops_arg arg0, ops_arg arg1) {
#else
void ops_par_loop_tvd_kernel_execute(ops_kernel_descriptor *desc) {
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
    if (!ops_checkpointing_before(args, 2, range, 9)) return;
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timing_realloc(block->instance, 9, "tvd_kernel");
        block->instance->OPS_kernels[9].count++;
        ops_timers_core(&__c1, &__t1);
    }

//  =================================================
//  compute locally allocated range for the sub-block
//  =================================================
    int start_indx[1];
    int end_indx[1];
#ifdef OPS_MPI
    int arg_idx[1];
#endif

#if defined(OPS_LAZY) || !defined(OPS_MPI)
    for (int n = 0; n < 1; n++) {
        start_indx[n] = range[2*n];
        end_indx[n]   = range[2*n+1];
    }
#else
    if (compute_ranges(args, 2, block, range, start_indx, end_indx, arg_idx) < 0) return;
#endif

    int xdim0 = args[0].dat->size[0];
    int xdim1 = args[1].dat->size[0];

    if (xdim0 != dims_tvd_kernel_h[0][0] || xdim1 != dims_tvd_kernel_h[1][0]) {
        dims_tvd_kernel_h[0][0] = xdim0;
        dims_tvd_kernel_h[1][0] = xdim1;

        cutilSafeCall(block->instance->ostream(), cudaMemcpyToSymbol( dims_tvd_kernel, dims_tvd_kernel_h, sizeof(dims_tvd_kernel)));
    }


    int x_size = MAX(0,end_indx[0]-start_indx[0]);

    dim3 grid( (x_size-1)/block->instance->OPS_block_size_x+ 1, 1, 1);
    dim3 tblock(block->instance->OPS_block_size_x,1,1);



    long long int dat0 = (block->instance->OPS_soa ? args[0].dat->type_size : args[0].dat->elem_size);
    long long int dat1 = (block->instance->OPS_soa ? args[1].dat->type_size : args[1].dat->elem_size);

    char *p_a[2];

//  =======================
//  set up initial pointers
//  =======================
    long long int base0 = args[0].dat->base_offset + 
             dat0 * 1 * (start_indx[0] * args[0].stencil->stride[0]);
    p_a[0] = (char *)args[0].data_d + base0;

    long long int base1 = args[1].dat->base_offset + 
             dat1 * 1 * (start_indx[0] * args[1].stencil->stride[0]);
    p_a[1] = (char *)args[1].data_d + base1;

//  =============
//  Halo exchange
//  =============
#ifndef OPS_LAZY
    ops_H_D_exchanges_device(args, 2);
    ops_halo_exchanges(args, 2, range);
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[9].mpi_time += __t2 - __t1;
    }


//call kernel wrapper function, passing in pointers to data
    if (x_size > 0) {

        ops_tvd_kernel<<<grid, tblock >>> ( 
         (double *)p_a[0], (double *)p_a[1],
        x_size);

    }

    cutilSafeCall(block->instance->ostream(), cudaGetLastError());

    if (block->instance->OPS_diags>1) {
        cutilSafeCall(block->instance->ostream(), cudaDeviceSynchronize());
        ops_timers_core(&__c1, &__t1);
        block->instance->OPS_kernels[9].time += __t1 - __t2;
    }

#ifndef OPS_LAZY
    ops_set_dirtybit_device(args, 2);
    ops_set_halo_dirtybit3(&args[1], range);
#endif

    if (block->instance->OPS_diags > 1) {
        //  ====================
        //  Update kernel record
        //  ====================
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[9].mpi_time += __t2 - __t1;
        block->instance->OPS_kernels[9].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg0);
        block->instance->OPS_kernels[9].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg1);
    }
}

#ifdef OPS_LAZY
void ops_par_loop_tvd_kernel(char const *name, ops_block block, int dim, int* range,
 ops_arg arg0, ops_arg arg1)
{
    ops_arg args[2];

    args[0] = arg0;
    args[1] = arg1;

    create_kerneldesc_and_enque("tvd_kernel", args, 2, 9, dim, 1, range, block, ops_par_loop_tvd_kernel_execute);
}
#endif
