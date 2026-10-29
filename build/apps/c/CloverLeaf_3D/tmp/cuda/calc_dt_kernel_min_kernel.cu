// Auto-generated at 2026-10-29 00:13:59.470383 by ops-translator legacy

__constant__ int dims_calc_dt_kernel_min [2][2];
static int dims_calc_dt_kernel_min_h [2][2] = {{0}};

//  =============
//  User function
//  =============
__device__ void calc_dt_kernel_min_gpu(const ACC<double>& dt_min,
  double* dt_min_val) {
  *dt_min_val = MIN(*dt_min_val, dt_min(0,0,0));

}


//  ============================
//  Cuda kernel wrapper function
//  ============================
__global__ void ops_calc_dt_kernel_min(
double* __restrict arg0,
double* __restrict arg1,
int size0,
int size1,
int size2) {
    double arg1_l[1];
    for (int d = 0; d < 1; d++) arg1_l[d] = INFINITY_double;

    int idx_z = blockDim.z * blockIdx.z + threadIdx.z;
    int idx_y = blockDim.y * blockIdx.y + threadIdx.y;
    int idx_x = blockDim.x * blockIdx.x + threadIdx.x;

    arg0 += idx_x * 1*1 + idx_y * 1*1 * dims_calc_dt_kernel_min[0][0] + idx_z * 1*1 * dims_calc_dt_kernel_min[0][0] * dims_calc_dt_kernel_min[0][1];

    if (idx_x < size0 && idx_y < size1 && idx_z < size2) {
        const ACC<double> argp0(dims_calc_dt_kernel_min[0][0], dims_calc_dt_kernel_min[0][1], arg0);
        calc_dt_kernel_min_gpu(
         argp0, arg1_l);
    }

//  ==============================
//  Reduction across thread blocks
//  ==============================
    for(int d = 0; d < 1; d++)
        ops_reduction_cuda<OPS_MIN>(&arg1[d+(blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y)*1],arg1_l[d]);

} //End of cuda kernel wrapper function

//  ==================
//  Host stub function
//  ==================
#ifndef OPS_LAZY
void ops_par_loop_calc_dt_kernel_min(char const *name, ops_block block, int dim, int* range,
 ops_arg arg0, ops_arg arg1) {
#else
void ops_par_loop_calc_dt_kernel_min_execute(ops_kernel_descriptor *desc) {
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
    if (!ops_checkpointing_before(args, 2, range, 34)) return;
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timing_realloc(block->instance, 34, "calc_dt_kernel_min");
        block->instance->OPS_kernels[34].count++;
        ops_timers_core(&__c1, &__t1);
    }

//  =================================================
//  compute locally allocated range for the sub-block
//  =================================================
    int start_indx[3];
    int end_indx[3];
#ifdef OPS_MPI
    int arg_idx[3];
#endif

#if defined(OPS_LAZY) || !defined(OPS_MPI)
    for (int n = 0; n < 3; n++) {
        start_indx[n] = range[2*n];
        end_indx[n]   = range[2*n+1];
    }
#else
    if (compute_ranges(args, 2, block, range, start_indx, end_indx, arg_idx) < 0) return;
#endif

    int xdim0 = args[0].dat->size[0];
    int ydim0 = args[0].dat->size[1];

    if (xdim0 != dims_calc_dt_kernel_min_h[0][0] || ydim0 != dims_calc_dt_kernel_min_h[0][1]) {
        dims_calc_dt_kernel_min_h[0][0] = xdim0;
        dims_calc_dt_kernel_min_h[0][1] = ydim0;

        cutilSafeCall(block->instance->ostream(), cudaMemcpyToSymbol( dims_calc_dt_kernel_min, dims_calc_dt_kernel_min_h, sizeof(dims_calc_dt_kernel_min)));
    }

#ifdef OPS_MPI
    double *arg1h = (double *)(((ops_reduction)args[1].data)->data + ((ops_reduction)args[1].data)->size * block->index);
#else
    double *arg1h = (double *)(((ops_reduction)args[1].data)->data);
#endif

    int x_size = MAX(0,end_indx[0]-start_indx[0]);
    int y_size = MAX(0,end_indx[1]-start_indx[1]);
    int z_size = MAX(0,end_indx[2]-start_indx[2]);

    dim3 grid( (x_size-1)/block->instance->OPS_block_size_x+ 1, (y_size-1)/block->instance->OPS_block_size_y + 1, (z_size-1)/block->instance->OPS_block_size_z +1);
    dim3 tblock(block->instance->OPS_block_size_x,block->instance->OPS_block_size_y,block->instance->OPS_block_size_z);

    int nblocks = ((x_size-1)/block->instance->OPS_block_size_x+ 1)*((y_size-1)/block->instance->OPS_block_size_y + 1)*((z_size-1)/block->instance->OPS_block_size_z +1);

    int maxblocks = nblocks;
    int reduct_bytes = 0;
    size_t reduct_size = 0;

    reduct_bytes += ROUND_UP(maxblocks*1*sizeof(double));
    reduct_size = MAX(reduct_size,1*sizeof(double));

    reallocReductArrays(block->instance,reduct_bytes);
    reduct_bytes = 0;

    arg1.data = block->instance->OPS_reduct_h + reduct_bytes;
    arg1.data_d = block->instance->OPS_reduct_d + reduct_bytes;
    for (int b = 0; b < maxblocks; b++) {
      for (int d = 0; d < 1; d++)   ((double *)arg1.data)[d+b*1] = INFINITY_double;
    }
    reduct_bytes += ROUND_UP(maxblocks*1*sizeof(double));

    mvReductArraysToDevice(block->instance, reduct_bytes);

    long long int dat0 = (block->instance->OPS_soa ? args[0].dat->type_size : args[0].dat->elem_size);

    char *p_a[2];

//  =======================
//  set up initial pointers
//  =======================
    long long int base0 = args[0].dat->base_offset + 
             dat0 * 1 * (start_indx[0] * args[0].stencil->stride[0]);
    base0 = base0 + dat0 *
                       args[0].dat->size[0] *
                      (start_indx[1] * args[0].stencil->stride[1]);
    base0 = base0 + dat0 *
                       args[0].dat->size[0] *
                       args[0].dat->size[1] *
                      (start_indx[2] * args[0].stencil->stride[2]);
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
        block->instance->OPS_kernels[34].mpi_time += __t2 - __t1;
    }

    size_t nshared = 0;
    int nthread = block->instance->OPS_block_size_x*block->instance->OPS_block_size_y*block->instance->OPS_block_size_z;

    nshared = MAX(nshared,sizeof(double)*1);

    nshared = MAX(nshared*nthread,reduct_size*nthread);

//call kernel wrapper function, passing in pointers to data
    if (x_size > 0 && y_size > 0 && z_size > 0) {

        ops_calc_dt_kernel_min<<<grid, tblock, nshared >>> ( 
         (double *)p_a[0], (double *)arg1.data_d,
        x_size, y_size, z_size);

    }

    cutilSafeCall(block->instance->ostream(), cudaGetLastError());

    mvReductArraysToHost(block->instance,reduct_bytes);
    for (int b = 0; b < maxblocks; b++) {
        for (int d = 0; d < 1; d++) {
            arg1h[d] = MIN(arg1h[d],((double *)arg1.data)[d+b*1]);
        }
    }
    arg1.data = (char *)arg1h;

    if (block->instance->OPS_diags>1) {
        cutilSafeCall(block->instance->ostream(), cudaDeviceSynchronize());
        ops_timers_core(&__c1, &__t1);
        block->instance->OPS_kernels[34].time += __t1 - __t2;
    }

#ifndef OPS_LAZY
    ops_set_dirtybit_device(args, 2);
#endif

    if (block->instance->OPS_diags > 1) {
        //  ====================
        //  Update kernel record
        //  ====================
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[34].mpi_time += __t2 - __t1;
        block->instance->OPS_kernels[34].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg0);
    }
}

#ifdef OPS_LAZY
void ops_par_loop_calc_dt_kernel_min(char const *name, ops_block block, int dim, int* range,
 ops_arg arg0, ops_arg arg1)
{
    ops_arg args[2];

    args[0] = arg0;
    args[1] = arg1;

    create_kerneldesc_and_enque("calc_dt_kernel_min", args, 2, 34, dim, 1, range, block, ops_par_loop_calc_dt_kernel_min_execute);
}
#endif
