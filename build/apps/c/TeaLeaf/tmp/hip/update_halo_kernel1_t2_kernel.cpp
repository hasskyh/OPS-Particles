// Auto-generated at 2026-10-29 00:14:10.066440 by ops-translator legacy

__constant__ int dims_update_halo_kernel1_t2 [7][1];
static int dims_update_halo_kernel1_t2_h [7][1] = {{0}};

//  =============
//  User function
//  =============
__device__ inline void update_halo_kernel1_t2_gpu(ACC<double> &density0,
  ACC<double> &energy0,
  ACC<double> &energy1,
  ACC<double> &u,
  ACC<double> &p,
  ACC<double> &sd,
  const int* fields) {
  if(fields[FIELD_DENSITY] == 1) density0(0,0) = density0(0,-3);
  if(fields[FIELD_ENERGY0] == 1) energy0(0,0) = energy0(0,-3);
  if(fields[FIELD_ENERGY1] == 1) energy1(0,0) = energy1(0,-3);
  if(fields[FIELD_U] == 1) u(0,0) = u(0,-3);
  if(fields[FIELD_P] == 1) p(0,0) = p(0,-3);
  if(fields[FIELD_SD] == 1) sd(0,0) = sd(0,-3);

}


//  ============================
//  Cuda kernel wrapper function
//  ============================
__global__ void ops_update_halo_kernel1_t2(
double* __restrict arg0,
double* __restrict arg1,
double* __restrict arg2,
double* __restrict arg3,
double* __restrict arg4,
double* __restrict arg5,
const int* __restrict arg6,
int size0,
int size1) {
    int idx_y = blockDim.y * blockIdx.y + threadIdx.y;
    int idx_x = blockDim.x * blockIdx.x + threadIdx.x;

    arg0 += idx_x * 1*1 + idx_y * 1*1 * dims_update_halo_kernel1_t2[0][0];
    arg1 += idx_x * 1*1 + idx_y * 1*1 * dims_update_halo_kernel1_t2[1][0];
    arg2 += idx_x * 1*1 + idx_y * 1*1 * dims_update_halo_kernel1_t2[2][0];
    arg3 += idx_x * 1*1 + idx_y * 1*1 * dims_update_halo_kernel1_t2[3][0];
    arg4 += idx_x * 1*1 + idx_y * 1*1 * dims_update_halo_kernel1_t2[4][0];
    arg5 += idx_x * 1*1 + idx_y * 1*1 * dims_update_halo_kernel1_t2[5][0];

    if (idx_x < size0 && idx_y < size1) {
        ACC<double> argp0(dims_update_halo_kernel1_t2[0][0], arg0);
        ACC<double> argp1(dims_update_halo_kernel1_t2[1][0], arg1);
        ACC<double> argp2(dims_update_halo_kernel1_t2[2][0], arg2);
        ACC<double> argp3(dims_update_halo_kernel1_t2[3][0], arg3);
        ACC<double> argp4(dims_update_halo_kernel1_t2[4][0], arg4);
        ACC<double> argp5(dims_update_halo_kernel1_t2[5][0], arg5);
        update_halo_kernel1_t2_gpu(
         argp0, argp1, argp2, argp3, argp4,
         argp5, arg6);
    }

} //End of cuda kernel wrapper function

//  ==================
//  Host stub function
//  ==================
#ifndef OPS_LAZY
void ops_par_loop_update_halo_kernel1_t2(char const *name, ops_block block, int dim, int* range,
 ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
 ops_arg arg4, ops_arg arg5, ops_arg arg6) {
#else
void ops_par_loop_update_halo_kernel1_t2_execute(ops_kernel_descriptor *desc) {
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
    if (!ops_checkpointing_before(args, 7, range, 41)) return;
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timing_realloc(block->instance, 41, "update_halo_kernel1_t2");
        block->instance->OPS_kernels[41].count++;
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

    if (xdim0 != dims_update_halo_kernel1_t2_h[0][0] || xdim1 != dims_update_halo_kernel1_t2_h[1][0] || xdim2 != dims_update_halo_kernel1_t2_h[2][0] || xdim3 != dims_update_halo_kernel1_t2_h[3][0] || xdim4 != dims_update_halo_kernel1_t2_h[4][0] || xdim5 != dims_update_halo_kernel1_t2_h[5][0]) {
        dims_update_halo_kernel1_t2_h[0][0] = xdim0;
        dims_update_halo_kernel1_t2_h[1][0] = xdim1;
        dims_update_halo_kernel1_t2_h[2][0] = xdim2;
        dims_update_halo_kernel1_t2_h[3][0] = xdim3;
        dims_update_halo_kernel1_t2_h[4][0] = xdim4;
        dims_update_halo_kernel1_t2_h[5][0] = xdim5;

        hipSafeCall(block->instance->ostream(), hipMemcpyToSymbol( dims_update_halo_kernel1_t2, dims_update_halo_kernel1_t2_h, sizeof(dims_update_halo_kernel1_t2)));
    }

    int *arg6h = (int *)arg6.data;

    int x_size = MAX(0,end_indx[0]-start_indx[0]);
    int y_size = MAX(0,end_indx[1]-start_indx[1]);

    dim3 grid( (x_size-1)/block->instance->OPS_block_size_x+ 1, (y_size-1)/block->instance->OPS_block_size_y + 1, 1);
    dim3 tblock(block->instance->OPS_block_size_x,block->instance->OPS_block_size_y,block->instance->OPS_block_size_z);

    int consts_bytes = 0;

    consts_bytes += ROUND_UP(arg6.dim*sizeof(int));

    reallocConstArrays(block->instance,consts_bytes);
    consts_bytes = 0;
    arg6.data = block->instance->OPS_consts_h + consts_bytes;
    arg6.data_d = block->instance->OPS_consts_d + consts_bytes;
    for (int d=0; d<arg6.dim; d++) ((int *)arg6.data)[d] = arg6h[d];
    consts_bytes += ROUND_UP(arg6.dim*sizeof(int));
    mvConstArraysToDevice(block->instance,consts_bytes);

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
        block->instance->OPS_kernels[41].mpi_time += __t2 - __t1;
    }


//call kernel wrapper function, passing in pointers to data
    if (x_size > 0 && y_size > 0) {

        ops_update_halo_kernel1_t2<<<grid, tblock >>> ( 
         (double *)p_a[0], (double *)p_a[1],
         (double *)p_a[2], (double *)p_a[3],
         (double *)p_a[4], (double *)p_a[5],
         (int *)arg6.data_d,
        x_size, y_size);

    }

    hipSafeCall(block->instance->ostream(), hipGetLastError());

    if (block->instance->OPS_diags>1) {
        hipSafeCall(block->instance->ostream(), hipDeviceSynchronize());
        ops_timers_core(&__c1, &__t1);
        block->instance->OPS_kernels[41].time += __t1 - __t2;
    }

#ifndef OPS_LAZY
    ops_set_dirtybit_device(args, 7);
    ops_set_halo_dirtybit3(&args[0], range);
    ops_set_halo_dirtybit3(&args[1], range);
    ops_set_halo_dirtybit3(&args[2], range);
    ops_set_halo_dirtybit3(&args[3], range);
    ops_set_halo_dirtybit3(&args[4], range);
    ops_set_halo_dirtybit3(&args[5], range);
#endif

    if (block->instance->OPS_diags > 1) {
        //  ====================
        //  Update kernel record
        //  ====================
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[41].mpi_time += __t2 - __t1;
        block->instance->OPS_kernels[41].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg0);
        block->instance->OPS_kernels[41].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg1);
        block->instance->OPS_kernels[41].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg2);
        block->instance->OPS_kernels[41].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg3);
        block->instance->OPS_kernels[41].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg4);
        block->instance->OPS_kernels[41].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg5);
    }
}

#ifdef OPS_LAZY
void ops_par_loop_update_halo_kernel1_t2(char const *name, ops_block block, int dim, int* range,
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

    create_kerneldesc_and_enque("update_halo_kernel1_t2", args, 7, 41, dim, 1, range, block, ops_par_loop_update_halo_kernel1_t2_execute);
}
#endif
