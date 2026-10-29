// Auto-generated at 2026-10-29 00:14:11.461687 by ops-translator legacy

__constant__ int dims_updateRK3_kernel [11][1];
static int dims_updateRK3_kernel_h [11][1] = {{0}};

//  =============
//  User function
//  =============
__device__ void updateRK3_kernel_gpu(ACC<double> &rho_new,
  ACC<double>& rhou_new,
  ACC<double>& rhoE_new,
  ACC<double> &rho_old,
  ACC<double>& rhou_old,
  ACC<double>& rhoE_old,
  ACC<double> &rho_res,
  ACC<double> &rhou_res,
  ACC<double> &rhoE_res,
  const double* a1,
  const double* a2) {

  rho_new(0) = rho_old(0) + dt * a1[0] * (-rho_res(0));
  rhou_new(0) = rhou_old(0) + dt * a1[0] * (-rhou_res(0));
  rhoE_new(0) = rhoE_old(0) + dt * a1[0] * (-rhoE_res(0));

  rho_old(0) = rho_old(0) + dt * a2[0] * (-rho_res(0));
  rhou_old(0) = rhou_old(0) + dt * a2[0] * (-rhou_res(0));
  rhoE_old(0) = rhoE_old(0) + dt * a2[0] * (-rhoE_res(0));
  rho_res(0) = 0;
  rhou_res(0) = 0;
  rhoE_res(0) = 0;
  }


//  ============================
//  Cuda kernel wrapper function
//  ============================
__global__ void ops_updateRK3_kernel(
double* __restrict arg0,
double* __restrict arg1,
double* __restrict arg2,
double* __restrict arg3,
double* __restrict arg4,
double* __restrict arg5,
double* __restrict arg6,
double* __restrict arg7,
double* __restrict arg8,
const double arg9,
const double arg10,
int size0) {
    int idx_x = blockDim.x * blockIdx.x + threadIdx.x;

    arg0 += idx_x * 1*1;
    arg1 += idx_x * 1*1;
    arg2 += idx_x * 1*1;
    arg3 += idx_x * 1*1;
    arg4 += idx_x * 1*1;
    arg5 += idx_x * 1*1;
    arg6 += idx_x * 1*1;
    arg7 += idx_x * 1*1;
    arg8 += idx_x * 1*1;

    if (idx_x < size0) {
        ACC<double> argp0(arg0);
        ACC<double> argp1(arg1);
        ACC<double> argp2(arg2);
        ACC<double> argp3(arg3);
        ACC<double> argp4(arg4);
        ACC<double> argp5(arg5);
        ACC<double> argp6(arg6);
        ACC<double> argp7(arg7);
        ACC<double> argp8(arg8);
        updateRK3_kernel_gpu(
         argp0, argp1, argp2, argp3, argp4,
         argp5, argp6, argp7, argp8, &arg9,
         &arg10);
    }

} //End of cuda kernel wrapper function

//  ==================
//  Host stub function
//  ==================
#ifndef OPS_LAZY
void ops_par_loop_updateRK3_kernel(char const *name, ops_block block, int dim, int* range,
 ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
 ops_arg arg4, ops_arg arg5, ops_arg arg6, ops_arg arg7,
 ops_arg arg8, ops_arg arg9, ops_arg arg10) {
#else
void ops_par_loop_updateRK3_kernel_execute(ops_kernel_descriptor *desc) {
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
    ops_arg arg8 = desc->args[8];
    ops_arg arg9 = desc->args[9];
    ops_arg arg10 = desc->args[10];
#endif

//  ======
//  Timing
//  ======
    double __t1, __t2, __c1, __c2;

    ops_arg args[11];

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;
    args[3] = arg3;
    args[4] = arg4;
    args[5] = arg5;
    args[6] = arg6;
    args[7] = arg7;
    args[8] = arg8;
    args[9] = arg9;
    args[10] = arg10;

#if CHECKPOINTING && !OPS_LAZY
    if (!ops_checkpointing_before(args, 11, range, 6)) return;
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timing_realloc(block->instance, 6, "updateRK3_kernel");
        block->instance->OPS_kernels[6].count++;
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
    if (compute_ranges(args, 11, block, range, start_indx, end_indx, arg_idx) < 0) return;
#endif

    int xdim0 = args[0].dat->size[0];
    int xdim1 = args[1].dat->size[0];
    int xdim2 = args[2].dat->size[0];
    int xdim3 = args[3].dat->size[0];
    int xdim4 = args[4].dat->size[0];
    int xdim5 = args[5].dat->size[0];
    int xdim6 = args[6].dat->size[0];
    int xdim7 = args[7].dat->size[0];
    int xdim8 = args[8].dat->size[0];

    if (xdim0 != dims_updateRK3_kernel_h[0][0] || xdim1 != dims_updateRK3_kernel_h[1][0] || xdim2 != dims_updateRK3_kernel_h[2][0] || xdim3 != dims_updateRK3_kernel_h[3][0] || xdim4 != dims_updateRK3_kernel_h[4][0] || xdim5 != dims_updateRK3_kernel_h[5][0] || xdim6 != dims_updateRK3_kernel_h[6][0] || xdim7 != dims_updateRK3_kernel_h[7][0] || xdim8 != dims_updateRK3_kernel_h[8][0]) {
        dims_updateRK3_kernel_h[0][0] = xdim0;
        dims_updateRK3_kernel_h[1][0] = xdim1;
        dims_updateRK3_kernel_h[2][0] = xdim2;
        dims_updateRK3_kernel_h[3][0] = xdim3;
        dims_updateRK3_kernel_h[4][0] = xdim4;
        dims_updateRK3_kernel_h[5][0] = xdim5;
        dims_updateRK3_kernel_h[6][0] = xdim6;
        dims_updateRK3_kernel_h[7][0] = xdim7;
        dims_updateRK3_kernel_h[8][0] = xdim8;

        cutilSafeCall(block->instance->ostream(), cudaMemcpyToSymbol( dims_updateRK3_kernel, dims_updateRK3_kernel_h, sizeof(dims_updateRK3_kernel)));
    }


    int x_size = MAX(0,end_indx[0]-start_indx[0]);

    dim3 grid( (x_size-1)/block->instance->OPS_block_size_x+ 1, 1, 1);
    dim3 tblock(block->instance->OPS_block_size_x,1,1);



    long long int dat0 = (block->instance->OPS_soa ? args[0].dat->type_size : args[0].dat->elem_size);
    long long int dat1 = (block->instance->OPS_soa ? args[1].dat->type_size : args[1].dat->elem_size);
    long long int dat2 = (block->instance->OPS_soa ? args[2].dat->type_size : args[2].dat->elem_size);
    long long int dat3 = (block->instance->OPS_soa ? args[3].dat->type_size : args[3].dat->elem_size);
    long long int dat4 = (block->instance->OPS_soa ? args[4].dat->type_size : args[4].dat->elem_size);
    long long int dat5 = (block->instance->OPS_soa ? args[5].dat->type_size : args[5].dat->elem_size);
    long long int dat6 = (block->instance->OPS_soa ? args[6].dat->type_size : args[6].dat->elem_size);
    long long int dat7 = (block->instance->OPS_soa ? args[7].dat->type_size : args[7].dat->elem_size);
    long long int dat8 = (block->instance->OPS_soa ? args[8].dat->type_size : args[8].dat->elem_size);

    char *p_a[11];

//  =======================
//  set up initial pointers
//  =======================
    long long int base0 = args[0].dat->base_offset + 
             dat0 * 1 * (start_indx[0] * args[0].stencil->stride[0]);
    p_a[0] = (char *)args[0].data_d + base0;

    long long int base1 = args[1].dat->base_offset + 
             dat1 * 1 * (start_indx[0] * args[1].stencil->stride[0]);
    p_a[1] = (char *)args[1].data_d + base1;

    long long int base2 = args[2].dat->base_offset + 
             dat2 * 1 * (start_indx[0] * args[2].stencil->stride[0]);
    p_a[2] = (char *)args[2].data_d + base2;

    long long int base3 = args[3].dat->base_offset + 
             dat3 * 1 * (start_indx[0] * args[3].stencil->stride[0]);
    p_a[3] = (char *)args[3].data_d + base3;

    long long int base4 = args[4].dat->base_offset + 
             dat4 * 1 * (start_indx[0] * args[4].stencil->stride[0]);
    p_a[4] = (char *)args[4].data_d + base4;

    long long int base5 = args[5].dat->base_offset + 
             dat5 * 1 * (start_indx[0] * args[5].stencil->stride[0]);
    p_a[5] = (char *)args[5].data_d + base5;

    long long int base6 = args[6].dat->base_offset + 
             dat6 * 1 * (start_indx[0] * args[6].stencil->stride[0]);
    p_a[6] = (char *)args[6].data_d + base6;

    long long int base7 = args[7].dat->base_offset + 
             dat7 * 1 * (start_indx[0] * args[7].stencil->stride[0]);
    p_a[7] = (char *)args[7].data_d + base7;

    long long int base8 = args[8].dat->base_offset + 
             dat8 * 1 * (start_indx[0] * args[8].stencil->stride[0]);
    p_a[8] = (char *)args[8].data_d + base8;

//  =============
//  Halo exchange
//  =============
#ifndef OPS_LAZY
    ops_H_D_exchanges_device(args, 11);
    ops_halo_exchanges(args, 11, range);
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[6].mpi_time += __t2 - __t1;
    }


//call kernel wrapper function, passing in pointers to data
    if (x_size > 0) {

        ops_updateRK3_kernel<<<grid, tblock >>> ( 
         (double *)p_a[0], (double *)p_a[1],
         (double *)p_a[2], (double *)p_a[3],
         (double *)p_a[4], (double *)p_a[5],
         (double *)p_a[6], (double *)p_a[7],
         (double *)p_a[8], *(double *)arg9.data,
         *(double *)arg10.data,
        x_size);

    }

    cutilSafeCall(block->instance->ostream(), cudaGetLastError());

    if (block->instance->OPS_diags>1) {
        cutilSafeCall(block->instance->ostream(), cudaDeviceSynchronize());
        ops_timers_core(&__c1, &__t1);
        block->instance->OPS_kernels[6].time += __t1 - __t2;
    }

#ifndef OPS_LAZY
    ops_set_dirtybit_device(args, 11);
    ops_set_halo_dirtybit3(&args[0], range);
    ops_set_halo_dirtybit3(&args[1], range);
    ops_set_halo_dirtybit3(&args[2], range);
    ops_set_halo_dirtybit3(&args[3], range);
    ops_set_halo_dirtybit3(&args[4], range);
    ops_set_halo_dirtybit3(&args[5], range);
    ops_set_halo_dirtybit3(&args[6], range);
    ops_set_halo_dirtybit3(&args[7], range);
    ops_set_halo_dirtybit3(&args[8], range);
#endif

    if (block->instance->OPS_diags > 1) {
        //  ====================
        //  Update kernel record
        //  ====================
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[6].mpi_time += __t2 - __t1;
        block->instance->OPS_kernels[6].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg0);
        block->instance->OPS_kernels[6].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg1);
        block->instance->OPS_kernels[6].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg2);
        block->instance->OPS_kernels[6].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg3);
        block->instance->OPS_kernels[6].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg4);
        block->instance->OPS_kernels[6].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg5);
        block->instance->OPS_kernels[6].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg6);
        block->instance->OPS_kernels[6].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg7);
        block->instance->OPS_kernels[6].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg8);
    }
}

#ifdef OPS_LAZY
void ops_par_loop_updateRK3_kernel(char const *name, ops_block block, int dim, int* range,
 ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
 ops_arg arg4, ops_arg arg5, ops_arg arg6, ops_arg arg7,
 ops_arg arg8, ops_arg arg9, ops_arg arg10)
{
    ops_arg args[11];

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;
    args[3] = arg3;
    args[4] = arg4;
    args[5] = arg5;
    args[6] = arg6;
    args[7] = arg7;
    args[8] = arg8;
    args[9] = arg9;
    args[10] = arg10;

    create_kerneldesc_and_enque("updateRK3_kernel", args, 11, 6, dim, 1, range, block, ops_par_loop_updateRK3_kernel_execute);
}
#endif
