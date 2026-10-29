// Auto-generated at 2026-10-29 00:13:50.433476 by ops-translator legacy

__constant__ int dims_calc_dt_kernel_print [7][1];
static int dims_calc_dt_kernel_print_h [7][1] = {{0}};

//  =============
//  User function
//  =============
__device__ void calc_dt_kernel_print_gpu(const ACC<double> &xvel0,
  const ACC<double> &yvel0,
  const ACC<double> &density0,
  const ACC<double> &energy0,
  const ACC<double> &pressure,
  const ACC<double> &soundspeed,
  double *output) {
  output[0] = xvel0(1,0);
  output[1] = yvel0(1,0);
  output[2] = xvel0(-1,0);
  output[3] = yvel0(-1,0);
  output[4] = xvel0(0,1);
  output[5] = yvel0(0,1);
  output[6] = xvel0(0,-1);
  output[7] = yvel0(0,-1);
  output[8] = density0(0,0);
  output[9] = energy0(0,0);
  output[10]= pressure(0,0);
  output[11]= soundspeed(0,0);

}


//  ============================
//  Cuda kernel wrapper function
//  ============================
__global__ void ops_calc_dt_kernel_print(
double* __restrict arg0,
double* __restrict arg1,
double* __restrict arg2,
double* __restrict arg3,
double* __restrict arg4,
double* __restrict arg5,
double* __restrict arg6,
int size0,
int size1) {
    double arg6_l[12];
    for (int d = 0; d < 12; d++) arg6_l[d] = ZERO_double;

    int idx_y = blockDim.y * blockIdx.y + threadIdx.y;
    int idx_x = blockDim.x * blockIdx.x + threadIdx.x;

    arg0 += idx_x * 1*1 + idx_y * 1*1 * dims_calc_dt_kernel_print[0][0];
    arg1 += idx_x * 1*1 + idx_y * 1*1 * dims_calc_dt_kernel_print[1][0];
    arg2 += idx_x * 1*1 + idx_y * 1*1 * dims_calc_dt_kernel_print[2][0];
    arg3 += idx_x * 1*1 + idx_y * 1*1 * dims_calc_dt_kernel_print[3][0];
    arg4 += idx_x * 1*1 + idx_y * 1*1 * dims_calc_dt_kernel_print[4][0];
    arg5 += idx_x * 1*1 + idx_y * 1*1 * dims_calc_dt_kernel_print[5][0];

    if (idx_x < size0 && idx_y < size1) {
        const ACC<double> argp0(dims_calc_dt_kernel_print[0][0], arg0);
        const ACC<double> argp1(dims_calc_dt_kernel_print[1][0], arg1);
        const ACC<double> argp2(dims_calc_dt_kernel_print[2][0], arg2);
        const ACC<double> argp3(dims_calc_dt_kernel_print[3][0], arg3);
        const ACC<double> argp4(dims_calc_dt_kernel_print[4][0], arg4);
        const ACC<double> argp5(dims_calc_dt_kernel_print[5][0], arg5);
        calc_dt_kernel_print_gpu(
         argp0, argp1, argp2, argp3, argp4,
         argp5, arg6_l);
    }

//  ==============================
//  Reduction across thread blocks
//  ==============================
    for(int d = 0; d < 12; d++)
        ops_reduction_hip<OPS_INC>(&arg6[d+(blockIdx.x + blockIdx.y*gridDim.x)*12],arg6_l[d]);

} //End of cuda kernel wrapper function

//  ==================
//  Host stub function
//  ==================
#ifndef OPS_LAZY
void ops_par_loop_calc_dt_kernel_print(char const *name, ops_block block, int dim, int* range,
 ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
 ops_arg arg4, ops_arg arg5, ops_arg arg6) {
#else
void ops_par_loop_calc_dt_kernel_print_execute(ops_kernel_descriptor *desc) {
    ops_block block = desc->block;
    int dim = desc->dim;
    int *range = desc->range;
    ops_arg arg0 = desc->args[0];
    ops_arg arg1 = desc->args[1];
    ops_arg arg2 = desc->args[2];
    ops_arg arg3 = desc->args[3];
    ops_arg arg4 = desc->args[4];
    ops_arg arg5 = desc->args[5];
    ops_arg arg6 = desc->args[6];
#endif

//  ======
//  Timing
//  ======
    double __t1, __t2, __c1, __c2;

    ops_arg args[7];

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;
    args[3] = arg3;
    args[4] = arg4;
    args[5] = arg5;
    args[6] = arg6;

#if CHECKPOINTING && !OPS_LAZY
    if (!ops_checkpointing_before(args, 7, range, 26)) return;
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timing_realloc(block->instance, 26, "calc_dt_kernel_print");
        block->instance->OPS_kernels[26].count++;
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
    if (compute_ranges(args, 7, block, range, start_indx, end_indx, arg_idx) < 0) return;
#endif

    int xdim0 = args[0].dat->size[0];
    int xdim1 = args[1].dat->size[0];
    int xdim2 = args[2].dat->size[0];
    int xdim3 = args[3].dat->size[0];
    int xdim4 = args[4].dat->size[0];
    int xdim5 = args[5].dat->size[0];

    if (xdim0 != dims_calc_dt_kernel_print_h[0][0] || xdim1 != dims_calc_dt_kernel_print_h[1][0] || xdim2 != dims_calc_dt_kernel_print_h[2][0] || xdim3 != dims_calc_dt_kernel_print_h[3][0] || xdim4 != dims_calc_dt_kernel_print_h[4][0] || xdim5 != dims_calc_dt_kernel_print_h[5][0]) {
        dims_calc_dt_kernel_print_h[0][0] = xdim0;
        dims_calc_dt_kernel_print_h[1][0] = xdim1;
        dims_calc_dt_kernel_print_h[2][0] = xdim2;
        dims_calc_dt_kernel_print_h[3][0] = xdim3;
        dims_calc_dt_kernel_print_h[4][0] = xdim4;
        dims_calc_dt_kernel_print_h[5][0] = xdim5;

        hipSafeCall(block->instance->ostream(), hipMemcpyToSymbol( dims_calc_dt_kernel_print, dims_calc_dt_kernel_print_h, sizeof(dims_calc_dt_kernel_print)));
    }

#ifdef OPS_MPI
    double *arg6h = (double *)(((ops_reduction)args[6].data)->data + ((ops_reduction)args[6].data)->size * block->index);
#else
    double *arg6h = (double *)(((ops_reduction)args[6].data)->data);
#endif

    int x_size = MAX(0,end_indx[0]-start_indx[0]);
    int y_size = MAX(0,end_indx[1]-start_indx[1]);

    dim3 grid( (x_size-1)/block->instance->OPS_block_size_x+ 1, (y_size-1)/block->instance->OPS_block_size_y + 1, 1);
    dim3 tblock(block->instance->OPS_block_size_x,block->instance->OPS_block_size_y,block->instance->OPS_block_size_z);

    int nblocks = ((x_size-1)/block->instance->OPS_block_size_x+ 1)*((y_size-1)/block->instance->OPS_block_size_y + 1);

    int maxblocks = nblocks;
    int reduct_bytes = 0;
    size_t reduct_size = 0;

    reduct_bytes += ROUND_UP(maxblocks*12*sizeof(double));
    reduct_size = MAX(reduct_size,12*sizeof(double));

    reallocReductArrays(block->instance,reduct_bytes);
    reduct_bytes = 0;

    arg6.data = block->instance->OPS_reduct_h + reduct_bytes;
    arg6.data_d = block->instance->OPS_reduct_d + reduct_bytes;
    for (int b = 0; b < maxblocks; b++) {
      for (int d = 0; d < 12; d++)   ((double *)arg6.data)[d+b*12] = ZERO_double;
    }
    reduct_bytes += ROUND_UP(maxblocks*12*sizeof(double));

    mvReductArraysToDevice(block->instance, reduct_bytes);

    long long int dat0 = (block->instance->OPS_soa ? args[0].dat->type_size : args[0].dat->elem_size);
    long long int dat1 = (block->instance->OPS_soa ? args[1].dat->type_size : args[1].dat->elem_size);
    long long int dat2 = (block->instance->OPS_soa ? args[2].dat->type_size : args[2].dat->elem_size);
    long long int dat3 = (block->instance->OPS_soa ? args[3].dat->type_size : args[3].dat->elem_size);
    long long int dat4 = (block->instance->OPS_soa ? args[4].dat->type_size : args[4].dat->elem_size);
    long long int dat5 = (block->instance->OPS_soa ? args[5].dat->type_size : args[5].dat->elem_size);

    char *p_a[7];

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
    ops_H_D_exchanges_device(args, 7);
    ops_halo_exchanges(args, 7, range);
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[26].mpi_time += __t2 - __t1;
    }

    size_t nshared = 0;
    int nthread = block->instance->OPS_block_size_x*block->instance->OPS_block_size_y*block->instance->OPS_block_size_z;

    nshared = MAX(nshared,sizeof(double)*12);

    nshared = MAX(nshared*nthread,reduct_size*nthread);

//call kernel wrapper function, passing in pointers to data
    if (x_size > 0 && y_size > 0) {

        ops_calc_dt_kernel_print<<<grid, tblock, nshared >>> ( 
         (double *)p_a[0], (double *)p_a[1],
         (double *)p_a[2], (double *)p_a[3],
         (double *)p_a[4], (double *)p_a[5],
         (double *)arg6.data_d,
        x_size, y_size);

    }

    hipSafeCall(block->instance->ostream(), hipGetLastError());

    mvReductArraysToHost(block->instance,reduct_bytes);
    for (int b = 0; b < maxblocks; b++) {
        for (int d = 0; d < 12; d++) {
            arg6h[d] = arg6h[d] + ((double *)arg6.data)[d+b*12];
        }
    }
    arg6.data = (char *)arg6h;

    if (block->instance->OPS_diags>1) {
        hipSafeCall(block->instance->ostream(), hipDeviceSynchronize());
        ops_timers_core(&__c1, &__t1);
        block->instance->OPS_kernels[26].time += __t1 - __t2;
    }

#ifndef OPS_LAZY
    ops_set_dirtybit_device(args, 7);
#endif

    if (block->instance->OPS_diags > 1) {
        //  ====================
        //  Update kernel record
        //  ====================
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[26].mpi_time += __t2 - __t1;
        block->instance->OPS_kernels[26].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg0);
        block->instance->OPS_kernels[26].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg1);
        block->instance->OPS_kernels[26].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg2);
        block->instance->OPS_kernels[26].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg3);
        block->instance->OPS_kernels[26].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg4);
        block->instance->OPS_kernels[26].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg5);
    }
}

#ifdef OPS_LAZY
void ops_par_loop_calc_dt_kernel_print(char const *name, ops_block block, int dim, int* range,
 ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
 ops_arg arg4, ops_arg arg5, ops_arg arg6)
{
    ops_arg args[7];

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;
    args[3] = arg3;
    args[4] = arg4;
    args[5] = arg5;
    args[6] = arg6;

    create_kerneldesc_and_enque("calc_dt_kernel_print", args, 7, 26, dim, 1, range, block, ops_par_loop_calc_dt_kernel_print_execute);
}
#endif
