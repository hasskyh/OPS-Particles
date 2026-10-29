// Auto-generated at 2026-10-29 00:14:10.047504 by ops-translator legacy

__constant__ int dims_tea_leaf_ppcg_inner2_kernel [5][1];
static int dims_tea_leaf_ppcg_inner2_kernel_h [5][1] = {{0}};

//  =============
//  User function
//  =============
__device__ void tea_leaf_ppcg_inner2_kernel_gpu(ACC<double> &sd,
  ACC<double> &utemp,
  const ACC<double> &z,
  const double *alpha,
  const double *beta) {
  sd(0,0) = (*alpha) * sd(0,0) + (*beta)*z(0,0);
  utemp(0,0) = utemp(0,0) + sd(0,0);
}


//  ============================
//  Cuda kernel wrapper function
//  ============================
__global__ void ops_tea_leaf_ppcg_inner2_kernel(
double* __restrict arg0,
double* __restrict arg1,
double* __restrict arg2,
const double arg3,
const double arg4,
int size0,
int size1) {
    int idx_y = blockDim.y * blockIdx.y + threadIdx.y;
    int idx_x = blockDim.x * blockIdx.x + threadIdx.x;

    arg0 += idx_x * 1*1 + idx_y * 1*1 * dims_tea_leaf_ppcg_inner2_kernel[0][0];
    arg1 += idx_x * 1*1 + idx_y * 1*1 * dims_tea_leaf_ppcg_inner2_kernel[1][0];
    arg2 += idx_x * 1*1 + idx_y * 1*1 * dims_tea_leaf_ppcg_inner2_kernel[2][0];

    if (idx_x < size0 && idx_y < size1) {
        ACC<double> argp0(dims_tea_leaf_ppcg_inner2_kernel[0][0], arg0);
        ACC<double> argp1(dims_tea_leaf_ppcg_inner2_kernel[1][0], arg1);
        const ACC<double> argp2(dims_tea_leaf_ppcg_inner2_kernel[2][0], arg2);
        tea_leaf_ppcg_inner2_kernel_gpu(
         argp0, argp1, argp2, &arg3, &arg4);
    }

} //End of cuda kernel wrapper function

//  ==================
//  Host stub function
//  ==================
#ifndef OPS_LAZY
void ops_par_loop_tea_leaf_ppcg_inner2_kernel(char const *name, ops_block block, int dim, int* range,
 ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
 ops_arg arg4) {
#else
void ops_par_loop_tea_leaf_ppcg_inner2_kernel_execute(ops_kernel_descriptor *desc) {
    ops_block block = desc->block;
    int dim = desc->dim;
    int *range = desc->range;
    ops_arg arg0 = desc->args[0];
    ops_arg arg1 = desc->args[1];
    ops_arg arg2 = desc->args[2];
    ops_arg arg3 = desc->args[3];
    ops_arg arg4 = desc->args[4];
#endif

//  ======
//  Timing
//  ======
    double __t1, __t2, __c1, __c2;

    ops_arg args[5];

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;
    args[3] = arg3;
    args[4] = arg4;

#if CHECKPOINTING && !OPS_LAZY
    if (!ops_checkpointing_before(args, 5, range, 37)) return;
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timing_realloc(block->instance, 37, "tea_leaf_ppcg_inner2_kernel");
        block->instance->OPS_kernels[37].count++;
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
    if (compute_ranges(args, 5, block, range, start_indx, end_indx, arg_idx) < 0) return;
#endif

    int xdim0 = args[0].dat->size[0];
    int xdim1 = args[1].dat->size[0];
    int xdim2 = args[2].dat->size[0];

    if (xdim0 != dims_tea_leaf_ppcg_inner2_kernel_h[0][0] || xdim1 != dims_tea_leaf_ppcg_inner2_kernel_h[1][0] || xdim2 != dims_tea_leaf_ppcg_inner2_kernel_h[2][0]) {
        dims_tea_leaf_ppcg_inner2_kernel_h[0][0] = xdim0;
        dims_tea_leaf_ppcg_inner2_kernel_h[1][0] = xdim1;
        dims_tea_leaf_ppcg_inner2_kernel_h[2][0] = xdim2;

        hipSafeCall(block->instance->ostream(), hipMemcpyToSymbol( dims_tea_leaf_ppcg_inner2_kernel, dims_tea_leaf_ppcg_inner2_kernel_h, sizeof(dims_tea_leaf_ppcg_inner2_kernel)));
    }


    int x_size = MAX(0,end_indx[0]-start_indx[0]);
    int y_size = MAX(0,end_indx[1]-start_indx[1]);

    dim3 grid( (x_size-1)/block->instance->OPS_block_size_x+ 1, (y_size-1)/block->instance->OPS_block_size_y + 1, 1);
    dim3 tblock(block->instance->OPS_block_size_x,block->instance->OPS_block_size_y,block->instance->OPS_block_size_z);



    long long int dat0 = (block->instance->OPS_soa ? args[0].dat->type_size : args[0].dat->elem_size);
    long long int dat1 = (block->instance->OPS_soa ? args[1].dat->type_size : args[1].dat->elem_size);
    long long int dat2 = (block->instance->OPS_soa ? args[2].dat->type_size : args[2].dat->elem_size);

    char *p_a[5];

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
    ops_H_D_exchanges_device(args, 5);
    ops_halo_exchanges(args, 5, range);
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[37].mpi_time += __t2 - __t1;
    }


//call kernel wrapper function, passing in pointers to data
    if (x_size > 0 && y_size > 0) {

        ops_tea_leaf_ppcg_inner2_kernel<<<grid, tblock >>> ( 
         (double *)p_a[0], (double *)p_a[1],
         (double *)p_a[2], *(double *)arg3.data,
         *(double *)arg4.data,
        x_size, y_size);

    }

    hipSafeCall(block->instance->ostream(), hipGetLastError());

    if (block->instance->OPS_diags>1) {
        hipSafeCall(block->instance->ostream(), hipDeviceSynchronize());
        ops_timers_core(&__c1, &__t1);
        block->instance->OPS_kernels[37].time += __t1 - __t2;
    }

#ifndef OPS_LAZY
    ops_set_dirtybit_device(args, 5);
    ops_set_halo_dirtybit3(&args[0], range);
    ops_set_halo_dirtybit3(&args[1], range);
#endif

    if (block->instance->OPS_diags > 1) {
        //  ====================
        //  Update kernel record
        //  ====================
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[37].mpi_time += __t2 - __t1;
        block->instance->OPS_kernels[37].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg0);
        block->instance->OPS_kernels[37].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg1);
        block->instance->OPS_kernels[37].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg2);
    }
}

#ifdef OPS_LAZY
void ops_par_loop_tea_leaf_ppcg_inner2_kernel(char const *name, ops_block block, int dim, int* range,
 ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
 ops_arg arg4)
{
    ops_arg args[5];

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;
    args[3] = arg3;
    args[4] = arg4;

    create_kerneldesc_and_enque("tea_leaf_ppcg_inner2_kernel", args, 5, 37, dim, 1, range, block, ops_par_loop_tea_leaf_ppcg_inner2_kernel_execute);
}
#endif
