// Auto-generated at 2026-10-29 00:14:09.356604 by ops-translator legacy

__constant__ int dims_tea_leaf_cg_calc_ur_r_reduce_kernel [4][1];
static int dims_tea_leaf_cg_calc_ur_r_reduce_kernel_h [4][1] = {{0}};

//  =============
//  User function
//  =============
__device__ void tea_leaf_cg_calc_ur_r_reduce_kernel_gpu(ACC<double> & r,
  const ACC<double> & w,
  const double * alpha,
  double *rnn) {
  r(0,0) = r(0,0) - (*alpha)*w(0,0);
  *rnn = *rnn +  r(0,0)*r(0,0);
}


//  ============================
//  Cuda kernel wrapper function
//  ============================
__global__ void ops_tea_leaf_cg_calc_ur_r_reduce_kernel(
double* __restrict arg0,
double* __restrict arg1,
const double arg2,
double* __restrict arg3,
int size0,
int size1) {
    double arg3_l[1];
    for (int d = 0; d < 1; d++) arg3_l[d] = ZERO_double;

    int idx_y = blockDim.y * blockIdx.y + threadIdx.y;
    int idx_x = blockDim.x * blockIdx.x + threadIdx.x;

    arg0 += idx_x * 1*1 + idx_y * 1*1 * dims_tea_leaf_cg_calc_ur_r_reduce_kernel[0][0];
    arg1 += idx_x * 1*1 + idx_y * 1*1 * dims_tea_leaf_cg_calc_ur_r_reduce_kernel[1][0];

    if (idx_x < size0 && idx_y < size1) {
        ACC<double> argp0(dims_tea_leaf_cg_calc_ur_r_reduce_kernel[0][0], arg0);
        const ACC<double> argp1(dims_tea_leaf_cg_calc_ur_r_reduce_kernel[1][0], arg1);
        tea_leaf_cg_calc_ur_r_reduce_kernel_gpu(
         argp0, argp1, &arg2, arg3_l);
    }

//  ==============================
//  Reduction across thread blocks
//  ==============================
    for(int d = 0; d < 1; d++)
        ops_reduction_cuda<OPS_INC>(&arg3[d+(blockIdx.x + blockIdx.y*gridDim.x)*1],arg3_l[d]);

} //End of cuda kernel wrapper function

//  ==================
//  Host stub function
//  ==================
#ifndef OPS_LAZY
void ops_par_loop_tea_leaf_cg_calc_ur_r_reduce_kernel(char const *name, ops_block block, int dim, int* range,
 ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3) {
#else
void ops_par_loop_tea_leaf_cg_calc_ur_r_reduce_kernel_execute(ops_kernel_descriptor *desc) {
    ops_block block = desc->block;
    int dim = desc->dim;
    int *range = desc->range;
    ops_arg arg0 = desc->args[0];
    ops_arg arg1 = desc->args[1];
    ops_arg arg2 = desc->args[2];
    ops_arg arg3 = desc->args[3];
#endif

//  ======
//  Timing
//  ======
    double __t1, __t2, __c1, __c2;

    ops_arg args[4];

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;
    args[3] = arg3;

#if CHECKPOINTING && !OPS_LAZY
    if (!ops_checkpointing_before(args, 4, range, 18)) return;
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timing_realloc(block->instance, 18, "tea_leaf_cg_calc_ur_r_reduce_kernel");
        block->instance->OPS_kernels[18].count++;
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
    if (compute_ranges(args, 4, block, range, start_indx, end_indx, arg_idx) < 0) return;
#endif

    int xdim0 = args[0].dat->size[0];
    int xdim1 = args[1].dat->size[0];

    if (xdim0 != dims_tea_leaf_cg_calc_ur_r_reduce_kernel_h[0][0] || xdim1 != dims_tea_leaf_cg_calc_ur_r_reduce_kernel_h[1][0]) {
        dims_tea_leaf_cg_calc_ur_r_reduce_kernel_h[0][0] = xdim0;
        dims_tea_leaf_cg_calc_ur_r_reduce_kernel_h[1][0] = xdim1;

        cutilSafeCall(block->instance->ostream(), cudaMemcpyToSymbol( dims_tea_leaf_cg_calc_ur_r_reduce_kernel, dims_tea_leaf_cg_calc_ur_r_reduce_kernel_h, sizeof(dims_tea_leaf_cg_calc_ur_r_reduce_kernel)));
    }

#ifdef OPS_MPI
    double *arg3h = (double *)(((ops_reduction)args[3].data)->data + ((ops_reduction)args[3].data)->size * block->index);
#else
    double *arg3h = (double *)(((ops_reduction)args[3].data)->data);
#endif

    int x_size = MAX(0,end_indx[0]-start_indx[0]);
    int y_size = MAX(0,end_indx[1]-start_indx[1]);

    dim3 grid( (x_size-1)/block->instance->OPS_block_size_x+ 1, (y_size-1)/block->instance->OPS_block_size_y + 1, 1);
    dim3 tblock(block->instance->OPS_block_size_x,block->instance->OPS_block_size_y,block->instance->OPS_block_size_z);

    int nblocks = ((x_size-1)/block->instance->OPS_block_size_x+ 1)*((y_size-1)/block->instance->OPS_block_size_y + 1);

    int maxblocks = nblocks;
    int reduct_bytes = 0;
    size_t reduct_size = 0;

    reduct_bytes += ROUND_UP(maxblocks*1*sizeof(double));
    reduct_size = MAX(reduct_size,1*sizeof(double));

    reallocReductArrays(block->instance,reduct_bytes);
    reduct_bytes = 0;

    arg3.data = block->instance->OPS_reduct_h + reduct_bytes;
    arg3.data_d = block->instance->OPS_reduct_d + reduct_bytes;
    for (int b = 0; b < maxblocks; b++) {
      for (int d = 0; d < 1; d++)   ((double *)arg3.data)[d+b*1] = ZERO_double;
    }
    reduct_bytes += ROUND_UP(maxblocks*1*sizeof(double));

    mvReductArraysToDevice(block->instance, reduct_bytes);

    long long int dat0 = (block->instance->OPS_soa ? args[0].dat->type_size : args[0].dat->elem_size);
    long long int dat1 = (block->instance->OPS_soa ? args[1].dat->type_size : args[1].dat->elem_size);

    char *p_a[4];

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

//  =============
//  Halo exchange
//  =============
#ifndef OPS_LAZY
    ops_H_D_exchanges_device(args, 4);
    ops_halo_exchanges(args, 4, range);
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[18].mpi_time += __t2 - __t1;
    }

    size_t nshared = 0;
    int nthread = block->instance->OPS_block_size_x*block->instance->OPS_block_size_y*block->instance->OPS_block_size_z;

    nshared = MAX(nshared,sizeof(double)*1);

    nshared = MAX(nshared*nthread,reduct_size*nthread);

//call kernel wrapper function, passing in pointers to data
    if (x_size > 0 && y_size > 0) {

        ops_tea_leaf_cg_calc_ur_r_reduce_kernel<<<grid, tblock, nshared >>> ( 
         (double *)p_a[0], (double *)p_a[1],
         *(double *)arg2.data, (double *)arg3.data_d,
        x_size, y_size);

    }

    cutilSafeCall(block->instance->ostream(), cudaGetLastError());

    mvReductArraysToHost(block->instance,reduct_bytes);
    for (int b = 0; b < maxblocks; b++) {
        for (int d = 0; d < 1; d++) {
            arg3h[d] = arg3h[d] + ((double *)arg3.data)[d+b*1];
        }
    }
    arg3.data = (char *)arg3h;

    if (block->instance->OPS_diags>1) {
        cutilSafeCall(block->instance->ostream(), cudaDeviceSynchronize());
        ops_timers_core(&__c1, &__t1);
        block->instance->OPS_kernels[18].time += __t1 - __t2;
    }

#ifndef OPS_LAZY
    ops_set_dirtybit_device(args, 4);
    ops_set_halo_dirtybit3(&args[0], range);
#endif

    if (block->instance->OPS_diags > 1) {
        //  ====================
        //  Update kernel record
        //  ====================
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[18].mpi_time += __t2 - __t1;
        block->instance->OPS_kernels[18].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg0);
        block->instance->OPS_kernels[18].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg1);
    }
}

#ifdef OPS_LAZY
void ops_par_loop_tea_leaf_cg_calc_ur_r_reduce_kernel(char const *name, ops_block block, int dim, int* range,
 ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3)
{
    ops_arg args[4];

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;
    args[3] = arg3;

    create_kerneldesc_and_enque("tea_leaf_cg_calc_ur_r_reduce_kernel", args, 4, 18, dim, 1, range, block, ops_par_loop_tea_leaf_cg_calc_ur_r_reduce_kernel_execute);
}
#endif
