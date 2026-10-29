// Auto-generated at 2026-10-29 00:13:59.829864 by ops-translator legacy

__constant__ int dims_revert_kernel [4][2];
static int dims_revert_kernel_h [4][2] = {{0}};

//  =============
//  User function
//  =============
__device__ void revert_kernel_gpu(const ACC<double> &density0,
  ACC<double> &density1,
  const ACC<double> &energy0,
  ACC<double> &energy1) {

  density1(0,0,0) = density0(0,0,0);
  energy1(0,0,0) = energy0(0,0,0);
}


//  ============================
//  Cuda kernel wrapper function
//  ============================
__global__ void ops_revert_kernel(
double* __restrict arg0,
double* __restrict arg1,
double* __restrict arg2,
double* __restrict arg3,
int size0,
int size1,
int size2) {
    int idx_z = blockDim.z * blockIdx.z + threadIdx.z;
    int idx_y = blockDim.y * blockIdx.y + threadIdx.y;
    int idx_x = blockDim.x * blockIdx.x + threadIdx.x;

    arg0 += idx_x * 1*1 + idx_y * 1*1 * dims_revert_kernel[0][0] + idx_z * 1*1 * dims_revert_kernel[0][0] * dims_revert_kernel[0][1];
    arg1 += idx_x * 1*1 + idx_y * 1*1 * dims_revert_kernel[1][0] + idx_z * 1*1 * dims_revert_kernel[1][0] * dims_revert_kernel[1][1];
    arg2 += idx_x * 1*1 + idx_y * 1*1 * dims_revert_kernel[2][0] + idx_z * 1*1 * dims_revert_kernel[2][0] * dims_revert_kernel[2][1];
    arg3 += idx_x * 1*1 + idx_y * 1*1 * dims_revert_kernel[3][0] + idx_z * 1*1 * dims_revert_kernel[3][0] * dims_revert_kernel[3][1];

    if (idx_x < size0 && idx_y < size1 && idx_z < size2) {
        const ACC<double> argp0(dims_revert_kernel[0][0], dims_revert_kernel[0][1], arg0);
        ACC<double> argp1(dims_revert_kernel[1][0], dims_revert_kernel[1][1], arg1);
        const ACC<double> argp2(dims_revert_kernel[2][0], dims_revert_kernel[2][1], arg2);
        ACC<double> argp3(dims_revert_kernel[3][0], dims_revert_kernel[3][1], arg3);
        revert_kernel_gpu(
         argp0, argp1, argp2, argp3);
    }

} //End of cuda kernel wrapper function

//  ==================
//  Host stub function
//  ==================
#ifndef OPS_LAZY
void ops_par_loop_revert_kernel(char const *name, ops_block block, int dim, int* range,
 ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3) {
#else
void ops_par_loop_revert_kernel_execute(ops_kernel_descriptor *desc) {
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
    if (!ops_checkpointing_before(args, 4, range, 55)) return;
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timing_realloc(block->instance, 55, "revert_kernel");
        block->instance->OPS_kernels[55].count++;
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
    if (compute_ranges(args, 4, block, range, start_indx, end_indx, arg_idx) < 0) return;
#endif

    int xdim0 = args[0].dat->size[0];
    int ydim0 = args[0].dat->size[1];
    int xdim1 = args[1].dat->size[0];
    int ydim1 = args[1].dat->size[1];
    int xdim2 = args[2].dat->size[0];
    int ydim2 = args[2].dat->size[1];
    int xdim3 = args[3].dat->size[0];
    int ydim3 = args[3].dat->size[1];

    if (xdim0 != dims_revert_kernel_h[0][0] || ydim0 != dims_revert_kernel_h[0][1] || xdim1 != dims_revert_kernel_h[1][0] || ydim1 != dims_revert_kernel_h[1][1] || xdim2 != dims_revert_kernel_h[2][0] || ydim2 != dims_revert_kernel_h[2][1] || xdim3 != dims_revert_kernel_h[3][0] || ydim3 != dims_revert_kernel_h[3][1]) {
        dims_revert_kernel_h[0][0] = xdim0;
        dims_revert_kernel_h[0][1] = ydim0;
        dims_revert_kernel_h[1][0] = xdim1;
        dims_revert_kernel_h[1][1] = ydim1;
        dims_revert_kernel_h[2][0] = xdim2;
        dims_revert_kernel_h[2][1] = ydim2;
        dims_revert_kernel_h[3][0] = xdim3;
        dims_revert_kernel_h[3][1] = ydim3;

        cutilSafeCall(block->instance->ostream(), cudaMemcpyToSymbol( dims_revert_kernel, dims_revert_kernel_h, sizeof(dims_revert_kernel)));
    }


    int x_size = MAX(0,end_indx[0]-start_indx[0]);
    int y_size = MAX(0,end_indx[1]-start_indx[1]);
    int z_size = MAX(0,end_indx[2]-start_indx[2]);

    dim3 grid( (x_size-1)/block->instance->OPS_block_size_x+ 1, (y_size-1)/block->instance->OPS_block_size_y + 1, (z_size-1)/block->instance->OPS_block_size_z +1);
    dim3 tblock(block->instance->OPS_block_size_x,block->instance->OPS_block_size_y,block->instance->OPS_block_size_z);



    long long int dat0 = (block->instance->OPS_soa ? args[0].dat->type_size : args[0].dat->elem_size);
    long long int dat1 = (block->instance->OPS_soa ? args[1].dat->type_size : args[1].dat->elem_size);
    long long int dat2 = (block->instance->OPS_soa ? args[2].dat->type_size : args[2].dat->elem_size);
    long long int dat3 = (block->instance->OPS_soa ? args[3].dat->type_size : args[3].dat->elem_size);

    char *p_a[4];

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

    long long int base1 = args[1].dat->base_offset + 
             dat1 * 1 * (start_indx[0] * args[1].stencil->stride[0]);
    base1 = base1 + dat1 *
                       args[1].dat->size[0] *
                      (start_indx[1] * args[1].stencil->stride[1]);
    base1 = base1 + dat1 *
                       args[1].dat->size[0] *
                       args[1].dat->size[1] *
                      (start_indx[2] * args[1].stencil->stride[2]);
    p_a[1] = (char *)args[1].data_d + base1;

    long long int base2 = args[2].dat->base_offset + 
             dat2 * 1 * (start_indx[0] * args[2].stencil->stride[0]);
    base2 = base2 + dat2 *
                       args[2].dat->size[0] *
                      (start_indx[1] * args[2].stencil->stride[1]);
    base2 = base2 + dat2 *
                       args[2].dat->size[0] *
                       args[2].dat->size[1] *
                      (start_indx[2] * args[2].stencil->stride[2]);
    p_a[2] = (char *)args[2].data_d + base2;

    long long int base3 = args[3].dat->base_offset + 
             dat3 * 1 * (start_indx[0] * args[3].stencil->stride[0]);
    base3 = base3 + dat3 *
                       args[3].dat->size[0] *
                      (start_indx[1] * args[3].stencil->stride[1]);
    base3 = base3 + dat3 *
                       args[3].dat->size[0] *
                       args[3].dat->size[1] *
                      (start_indx[2] * args[3].stencil->stride[2]);
    p_a[3] = (char *)args[3].data_d + base3;

//  =============
//  Halo exchange
//  =============
#ifndef OPS_LAZY
    ops_H_D_exchanges_device(args, 4);
    ops_halo_exchanges(args, 4, range);
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[55].mpi_time += __t2 - __t1;
    }


//call kernel wrapper function, passing in pointers to data
    if (x_size > 0 && y_size > 0 && z_size > 0) {

        ops_revert_kernel<<<grid, tblock >>> ( 
         (double *)p_a[0], (double *)p_a[1],
         (double *)p_a[2], (double *)p_a[3],
        x_size, y_size, z_size);

    }

    cutilSafeCall(block->instance->ostream(), cudaGetLastError());

    if (block->instance->OPS_diags>1) {
        cutilSafeCall(block->instance->ostream(), cudaDeviceSynchronize());
        ops_timers_core(&__c1, &__t1);
        block->instance->OPS_kernels[55].time += __t1 - __t2;
    }

#ifndef OPS_LAZY
    ops_set_dirtybit_device(args, 4);
    ops_set_halo_dirtybit3(&args[1], range);
    ops_set_halo_dirtybit3(&args[3], range);
#endif

    if (block->instance->OPS_diags > 1) {
        //  ====================
        //  Update kernel record
        //  ====================
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[55].mpi_time += __t2 - __t1;
        block->instance->OPS_kernels[55].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg0);
        block->instance->OPS_kernels[55].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg1);
        block->instance->OPS_kernels[55].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg2);
        block->instance->OPS_kernels[55].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg3);
    }
}

#ifdef OPS_LAZY
void ops_par_loop_revert_kernel(char const *name, ops_block block, int dim, int* range,
 ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3)
{
    ops_arg args[4];

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;
    args[3] = arg3;

    create_kerneldesc_and_enque("revert_kernel", args, 4, 55, dim, 1, range, block, ops_par_loop_revert_kernel_execute);
}
#endif
