// Auto-generated at 2026-10-29 00:13:46.485532 by ops-translator legacy

__constant__ int dims_poisson_kernel_populate [6][1];
static int dims_poisson_kernel_populate_h [6][1] = {{0}};

//  =============
//  User function
//  =============
__device__ void poisson_kernel_populate_gpu(const int *dispx,
  const int *dispy,
  const int *idx,
  ACC<double> &u,
  ACC<double> &f,
  ACC<double> &ref) {
  double x = dx * (double)(idx[0]+dispx[0]);
  double y = dy * (double)(idx[1]+dispy[0]);

  u(0,0) = myfun(sin(M_PI*x),cos(2.0*M_PI*y))-1.0;
  f(0,0) = -5.0*M_PI*M_PI*sin(M_PI*x)*cos(2.0*M_PI*y);
  ref(0,0) = sin(M_PI*x)*cos(2.0*M_PI*y);

}


//  ============================
//  Cuda kernel wrapper function
//  ============================
__global__ void ops_poisson_kernel_populate(
const int arg0,
const int arg1,
int arg_idx0, int arg_idx1,
double* __restrict arg3,
double* __restrict arg4,
double* __restrict arg5,
int size0,
int size1) {
    int idx_y = blockDim.y * blockIdx.y + threadIdx.y;
    int idx_x = blockDim.x * blockIdx.x + threadIdx.x;

    int arg_idx[2];
    arg_idx[0] = arg_idx0+idx_x;
    arg_idx[1] = arg_idx1+idx_y;

    arg3 += idx_x * 1*1 + idx_y * 1*1 * dims_poisson_kernel_populate[3][0];
    arg4 += idx_x * 1*1 + idx_y * 1*1 * dims_poisson_kernel_populate[4][0];
    arg5 += idx_x * 1*1 + idx_y * 1*1 * dims_poisson_kernel_populate[5][0];

    if (idx_x < size0 && idx_y < size1) {
        ACC<double> argp3(dims_poisson_kernel_populate[3][0], arg3);
        ACC<double> argp4(dims_poisson_kernel_populate[4][0], arg4);
        ACC<double> argp5(dims_poisson_kernel_populate[5][0], arg5);
        poisson_kernel_populate_gpu(
         &arg0, &arg1, arg_idx, argp3, argp4,
         argp5);
    }

} //End of cuda kernel wrapper function

//  ==================
//  Host stub function
//  ==================
#ifndef OPS_LAZY
void ops_par_loop_poisson_kernel_populate(char const *name, ops_block block, int dim, int* range,
 ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
 ops_arg arg4, ops_arg arg5) {
#else
void ops_par_loop_poisson_kernel_populate_execute(ops_kernel_descriptor *desc) {
    ops_block block = desc->block;
    int dim = desc->dim;
    int *range = desc->range;
    ops_arg arg0 = desc->args[0];
    ops_arg arg1 = desc->args[1];
    ops_arg arg2 = desc->args[2];
    ops_arg arg3 = desc->args[3];
    ops_arg arg4 = desc->args[4];
    ops_arg arg5 = desc->args[5];
#endif

//  ======
//  Timing
//  ======
    double __t1, __t2, __c1, __c2;

    ops_arg args[6];

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;
    args[3] = arg3;
    args[4] = arg4;
    args[5] = arg5;

#if CHECKPOINTING && !OPS_LAZY
    if (!ops_checkpointing_before(args, 6, range, 0)) return;
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timing_realloc(block->instance, 0, "poisson_kernel_populate");
        block->instance->OPS_kernels[0].count++;
        ops_timers_core(&__c1, &__t1);
    }

//  =================================================
//  compute locally allocated range for the sub-block
//  =================================================
    int start_indx[2];
    int end_indx[2];
    int arg_idx[2];

#if defined(OPS_LAZY) || !defined(OPS_MPI)
    for (int n = 0; n < 2; n++) {
        start_indx[n] = range[2*n];
        end_indx[n]   = range[2*n+1];
    }
#else
    if (compute_ranges(args, 6, block, range, start_indx, end_indx, arg_idx) < 0) return;
#endif

#if defined(OPS_MPI)
#if defined(OPS_LAZY)
    sub_block_list sb = OPS_sub_block_list[block->index];
    arg_idx[0] = sb->decomp_disp[0]+start_indx[0];
    arg_idx[1] = sb->decomp_disp[1]+start_indx[1];
#endif //OPS_LAZY
#else //OPS_MPI
    arg_idx[0] = start_indx[0];
    arg_idx[1] = start_indx[1];
#endif //OPS_MPI

    int xdim3 = args[3].dat->size[0];
    int xdim4 = args[4].dat->size[0];
    int xdim5 = args[5].dat->size[0];

    if (xdim3 != dims_poisson_kernel_populate_h[3][0] || xdim4 != dims_poisson_kernel_populate_h[4][0] || xdim5 != dims_poisson_kernel_populate_h[5][0]) {
        dims_poisson_kernel_populate_h[3][0] = xdim3;
        dims_poisson_kernel_populate_h[4][0] = xdim4;
        dims_poisson_kernel_populate_h[5][0] = xdim5;

        hipSafeCall(block->instance->ostream(), hipMemcpyToSymbol( dims_poisson_kernel_populate, dims_poisson_kernel_populate_h, sizeof(dims_poisson_kernel_populate)));
    }


    int x_size = MAX(0,end_indx[0]-start_indx[0]);
    int y_size = MAX(0,end_indx[1]-start_indx[1]);

    dim3 grid( (x_size-1)/block->instance->OPS_block_size_x+ 1, (y_size-1)/block->instance->OPS_block_size_y + 1, 1);
    dim3 tblock(block->instance->OPS_block_size_x,block->instance->OPS_block_size_y,block->instance->OPS_block_size_z);



    long long int dat3 = (block->instance->OPS_soa ? args[3].dat->type_size : args[3].dat->elem_size);
    long long int dat4 = (block->instance->OPS_soa ? args[4].dat->type_size : args[4].dat->elem_size);
    long long int dat5 = (block->instance->OPS_soa ? args[5].dat->type_size : args[5].dat->elem_size);

    char *p_a[6];

//  =======================
//  set up initial pointers
//  =======================
    long long int base3 = args[3].dat->base_offset + 
             dat3 * 1 * (start_indx[0] * args[3].stencil->stride[0]);
    base3 = base3 + dat3 *
                       args[3].dat->size[0] *
                      (start_indx[1] * args[3].stencil->stride[1]);
    p_a[3] = (char *)args[3].data_d + base3;

    long long int base4 = args[4].dat->base_offset + 
             dat4 * 1 * (start_indx[0] * args[4].stencil->stride[0]);
    base4 = base4 + dat4 *
                       args[4].dat->size[0] *
                      (start_indx[1] * args[4].stencil->stride[1]);
    p_a[4] = (char *)args[4].data_d + base4;

    long long int base5 = args[5].dat->base_offset + 
             dat5 * 1 * (start_indx[0] * args[5].stencil->stride[0]);
    base5 = base5 + dat5 *
                       args[5].dat->size[0] *
                      (start_indx[1] * args[5].stencil->stride[1]);
    p_a[5] = (char *)args[5].data_d + base5;

//  =============
//  Halo exchange
//  =============
#ifndef OPS_LAZY
    ops_H_D_exchanges_device(args, 6);
    ops_halo_exchanges(args, 6, range);
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[0].mpi_time += __t2 - __t1;
    }


//call kernel wrapper function, passing in pointers to data
    if (x_size > 0 && y_size > 0) {

        ops_poisson_kernel_populate<<<grid, tblock >>> ( 
         *(int *)arg0.data, *(int *)arg1.data,
         arg_idx[0], arg_idx[1], (double *)p_a[3],
         (double *)p_a[4], (double *)p_a[5],
        x_size, y_size);

    }

    hipSafeCall(block->instance->ostream(), hipGetLastError());

    if (block->instance->OPS_diags>1) {
        hipSafeCall(block->instance->ostream(), hipDeviceSynchronize());
        ops_timers_core(&__c1, &__t1);
        block->instance->OPS_kernels[0].time += __t1 - __t2;
    }

#ifndef OPS_LAZY
    ops_set_dirtybit_device(args, 6);
    ops_set_halo_dirtybit3(&args[3], range);
    ops_set_halo_dirtybit3(&args[4], range);
    ops_set_halo_dirtybit3(&args[5], range);
#endif

    if (block->instance->OPS_diags > 1) {
        //  ====================
        //  Update kernel record
        //  ====================
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[0].mpi_time += __t2 - __t1;
        block->instance->OPS_kernels[0].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg3);
        block->instance->OPS_kernels[0].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg4);
        block->instance->OPS_kernels[0].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg5);
    }
}

#ifdef OPS_LAZY
void ops_par_loop_poisson_kernel_populate(char const *name, ops_block block, int dim, int* range,
 ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
 ops_arg arg4, ops_arg arg5)
{
    ops_arg args[6];

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;
    args[3] = arg3;
    args[4] = arg4;
    args[5] = arg5;

    create_kerneldesc_and_enque("poisson_kernel_populate", args, 6, 0, dim, 1, range, block, ops_par_loop_poisson_kernel_populate_execute);
}
#endif
