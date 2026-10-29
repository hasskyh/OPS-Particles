// Auto-generated at 2026-10-29 00:13:49.611768 by ops-translator legacy

__constant__ int dims_update_halo_kernel1_b2 [8][1];
static int dims_update_halo_kernel1_b2_h [8][1] = {{0}};

//  =============
//  User function
//  =============
__device__ inline void update_halo_kernel1_b2_gpu(ACC<double> &density0,
  ACC<double> &density1,
  ACC<double> &energy0,
  ACC<double> &energy1,
  ACC<double> &pressure,
  ACC<double> &viscosity,
  ACC<double> &soundspeed,
  const int* fields) {
  if(fields[FIELD_DENSITY0] == 1) density0(0,0) = density0(0,3);
  if(fields[FIELD_DENSITY1] == 1) density1(0,0) = density1(0,3);
  if(fields[FIELD_ENERGY0] == 1) energy0(0,0) = energy0(0,3);
  if(fields[FIELD_ENERGY1] == 1) energy1(0,0) = energy1(0,3);
  if(fields[FIELD_PRESSURE] == 1) pressure(0,0) = pressure(0,3);
  if(fields[FIELD_VISCOSITY] == 1) viscosity(0,0) = viscosity(0,3);
  if(fields[FIELD_SOUNDSPEED] == 1) soundspeed(0,0) = soundspeed(0,3);

}


//  ============================
//  Cuda kernel wrapper function
//  ============================
__global__ void ops_update_halo_kernel1_b2(
double* __restrict arg0,
double* __restrict arg1,
double* __restrict arg2,
double* __restrict arg3,
double* __restrict arg4,
double* __restrict arg5,
double* __restrict arg6,
const int* __restrict arg7,
int size0,
int size1) {
    int idx_y = blockDim.y * blockIdx.y + threadIdx.y;
    int idx_x = blockDim.x * blockIdx.x + threadIdx.x;

    arg0 += idx_x * 1*1 + idx_y * 1*1 * dims_update_halo_kernel1_b2[0][0];
    arg1 += idx_x * 1*1 + idx_y * 1*1 * dims_update_halo_kernel1_b2[1][0];
    arg2 += idx_x * 1*1 + idx_y * 1*1 * dims_update_halo_kernel1_b2[2][0];
    arg3 += idx_x * 1*1 + idx_y * 1*1 * dims_update_halo_kernel1_b2[3][0];
    arg4 += idx_x * 1*1 + idx_y * 1*1 * dims_update_halo_kernel1_b2[4][0];
    arg5 += idx_x * 1*1 + idx_y * 1*1 * dims_update_halo_kernel1_b2[5][0];
    arg6 += idx_x * 1*1 + idx_y * 1*1 * dims_update_halo_kernel1_b2[6][0];

    if (idx_x < size0 && idx_y < size1) {
        ACC<double> argp0(dims_update_halo_kernel1_b2[0][0], arg0);
        ACC<double> argp1(dims_update_halo_kernel1_b2[1][0], arg1);
        ACC<double> argp2(dims_update_halo_kernel1_b2[2][0], arg2);
        ACC<double> argp3(dims_update_halo_kernel1_b2[3][0], arg3);
        ACC<double> argp4(dims_update_halo_kernel1_b2[4][0], arg4);
        ACC<double> argp5(dims_update_halo_kernel1_b2[5][0], arg5);
        ACC<double> argp6(dims_update_halo_kernel1_b2[6][0], arg6);
        update_halo_kernel1_b2_gpu(
         argp0, argp1, argp2, argp3, argp4,
         argp5, argp6, arg7);
    }

} //End of cuda kernel wrapper function

//  ==================
//  Host stub function
//  ==================
#ifndef OPS_LAZY
void ops_par_loop_update_halo_kernel1_b2(char const *name, ops_block block, int dim, int* range,
 ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
 ops_arg arg4, ops_arg arg5, ops_arg arg6, ops_arg arg7) {
#else
void ops_par_loop_update_halo_kernel1_b2_execute(ops_kernel_descriptor *desc) {
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
    ops_arg arg7 = desc->args[7];
#endif

//  ======
//  Timing
//  ======
    double __t1, __t2, __c1, __c2;

    ops_arg args[8];

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;
    args[3] = arg3;
    args[4] = arg4;
    args[5] = arg5;
    args[6] = arg6;
    args[7] = arg7;

#if CHECKPOINTING && !OPS_LAZY
    if (!ops_checkpointing_before(args, 8, range, 42)) return;
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timing_realloc(block->instance, 42, "update_halo_kernel1_b2");
        block->instance->OPS_kernels[42].count++;
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
    if (compute_ranges(args, 8, block, range, start_indx, end_indx, arg_idx) < 0) return;
#endif

    int xdim0 = args[0].dat->size[0];
    int xdim1 = args[1].dat->size[0];
    int xdim2 = args[2].dat->size[0];
    int xdim3 = args[3].dat->size[0];
    int xdim4 = args[4].dat->size[0];
    int xdim5 = args[5].dat->size[0];
    int xdim6 = args[6].dat->size[0];

    if (xdim0 != dims_update_halo_kernel1_b2_h[0][0] || xdim1 != dims_update_halo_kernel1_b2_h[1][0] || xdim2 != dims_update_halo_kernel1_b2_h[2][0] || xdim3 != dims_update_halo_kernel1_b2_h[3][0] || xdim4 != dims_update_halo_kernel1_b2_h[4][0] || xdim5 != dims_update_halo_kernel1_b2_h[5][0] || xdim6 != dims_update_halo_kernel1_b2_h[6][0]) {
        dims_update_halo_kernel1_b2_h[0][0] = xdim0;
        dims_update_halo_kernel1_b2_h[1][0] = xdim1;
        dims_update_halo_kernel1_b2_h[2][0] = xdim2;
        dims_update_halo_kernel1_b2_h[3][0] = xdim3;
        dims_update_halo_kernel1_b2_h[4][0] = xdim4;
        dims_update_halo_kernel1_b2_h[5][0] = xdim5;
        dims_update_halo_kernel1_b2_h[6][0] = xdim6;

        cutilSafeCall(block->instance->ostream(), cudaMemcpyToSymbol( dims_update_halo_kernel1_b2, dims_update_halo_kernel1_b2_h, sizeof(dims_update_halo_kernel1_b2)));
    }

    int *arg7h = (int *)arg7.data;

    int x_size = MAX(0,end_indx[0]-start_indx[0]);
    int y_size = MAX(0,end_indx[1]-start_indx[1]);

    dim3 grid( (x_size-1)/block->instance->OPS_block_size_x+ 1, (y_size-1)/block->instance->OPS_block_size_y + 1, 1);
    dim3 tblock(block->instance->OPS_block_size_x,block->instance->OPS_block_size_y,block->instance->OPS_block_size_z);

    int consts_bytes = 0;

    consts_bytes += ROUND_UP(arg7.dim*sizeof(int));

    reallocConstArrays(block->instance,consts_bytes);
    consts_bytes = 0;
    arg7.data = block->instance->OPS_consts_h + consts_bytes;
    arg7.data_d = block->instance->OPS_consts_d + consts_bytes;
    for (int d=0; d<arg7.dim; d++) ((int *)arg7.data)[d] = arg7h[d];
    consts_bytes += ROUND_UP(arg7.dim*sizeof(int));
    mvConstArraysToDevice(block->instance,consts_bytes);

    long long int dat0 = (block->instance->OPS_soa ? args[0].dat->type_size : args[0].dat->elem_size);
    long long int dat1 = (block->instance->OPS_soa ? args[1].dat->type_size : args[1].dat->elem_size);
    long long int dat2 = (block->instance->OPS_soa ? args[2].dat->type_size : args[2].dat->elem_size);
    long long int dat3 = (block->instance->OPS_soa ? args[3].dat->type_size : args[3].dat->elem_size);
    long long int dat4 = (block->instance->OPS_soa ? args[4].dat->type_size : args[4].dat->elem_size);
    long long int dat5 = (block->instance->OPS_soa ? args[5].dat->type_size : args[5].dat->elem_size);
    long long int dat6 = (block->instance->OPS_soa ? args[6].dat->type_size : args[6].dat->elem_size);

    char *p_a[8];

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

    long long int base6 = args[6].dat->base_offset + 
             dat6 * 1 * (start_indx[0] * args[6].stencil->stride[0]);
    base6 = base6 + dat6 *
                       args[6].dat->size[0] *
                      (start_indx[1] * args[6].stencil->stride[1]);
    p_a[6] = (char *)args[6].data_d + base6;

//  =============
//  Halo exchange
//  =============
#ifndef OPS_LAZY
    ops_H_D_exchanges_device(args, 8);
    ops_halo_exchanges(args, 8, range);
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[42].mpi_time += __t2 - __t1;
    }


//call kernel wrapper function, passing in pointers to data
    if (x_size > 0 && y_size > 0) {

        ops_update_halo_kernel1_b2<<<grid, tblock >>> ( 
         (double *)p_a[0], (double *)p_a[1],
         (double *)p_a[2], (double *)p_a[3],
         (double *)p_a[4], (double *)p_a[5],
         (double *)p_a[6], (int *)arg7.data_d,
        x_size, y_size);

    }

    cutilSafeCall(block->instance->ostream(), cudaGetLastError());

    if (block->instance->OPS_diags>1) {
        cutilSafeCall(block->instance->ostream(), cudaDeviceSynchronize());
        ops_timers_core(&__c1, &__t1);
        block->instance->OPS_kernels[42].time += __t1 - __t2;
    }

#ifndef OPS_LAZY
    ops_set_dirtybit_device(args, 8);
    ops_set_halo_dirtybit3(&args[0], range);
    ops_set_halo_dirtybit3(&args[1], range);
    ops_set_halo_dirtybit3(&args[2], range);
    ops_set_halo_dirtybit3(&args[3], range);
    ops_set_halo_dirtybit3(&args[4], range);
    ops_set_halo_dirtybit3(&args[5], range);
    ops_set_halo_dirtybit3(&args[6], range);
#endif

    if (block->instance->OPS_diags > 1) {
        //  ====================
        //  Update kernel record
        //  ====================
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[42].mpi_time += __t2 - __t1;
        block->instance->OPS_kernels[42].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg0);
        block->instance->OPS_kernels[42].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg1);
        block->instance->OPS_kernels[42].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg2);
        block->instance->OPS_kernels[42].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg3);
        block->instance->OPS_kernels[42].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg4);
        block->instance->OPS_kernels[42].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg5);
        block->instance->OPS_kernels[42].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg6);
    }
}

#ifdef OPS_LAZY
void ops_par_loop_update_halo_kernel1_b2(char const *name, ops_block block, int dim, int* range,
 ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
 ops_arg arg4, ops_arg arg5, ops_arg arg6, ops_arg arg7)
{
    ops_arg args[8];

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;
    args[3] = arg3;
    args[4] = arg4;
    args[5] = arg5;
    args[6] = arg6;
    args[7] = arg7;

    create_kerneldesc_and_enque("update_halo_kernel1_b2", args, 8, 42, dim, 1, range, block, ops_par_loop_update_halo_kernel1_b2_execute);
}
#endif
