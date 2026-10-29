// Auto-generated at 2026-10-29 00:14:13.207216 by ops-translator legacy

__constant__ int dims_KerComputeMacros [7][1];
static int dims_KerComputeMacros_h [7][1] = {{0}};

//  =============
//  User function
//  =============
__device__ void KerComputeMacros_gpu(ACC<double>& rho,
  ACC<double>& u,
  ACC<double>& v,
  const ACC<double>& f,
  const int *cx,
  const int *cy,
  const double *cs) {
  rho(0, 0) = 0.0;
  u(0, 0) = 0.0;
  v(0, 0) = 0.0;

  for (int i = 0; i < 9; i++) {
    rho(0, 0) += f(i, 0, 0);
    u(0, 0) += (*cs) * static_cast<double>(cx[i]) * f(i, 0, 0);
    v(0, 0) += (*cs) * cy[i] * f(i, 0, 0);
  }

  u(0, 0) /= rho(0, 0);
  v(0, 0) /= rho(0, 0);
}


//  ============================
//  Cuda kernel wrapper function
//  ============================
__global__ void ops_KerComputeMacros(
double* __restrict arg0,
double* __restrict arg1,
double* __restrict arg2,
double* __restrict arg3,
const int* __restrict arg4,
const int* __restrict arg5,
const double arg6,
int size0,
int size1) {
    int idx_y = blockDim.y * blockIdx.y + threadIdx.y;
    int idx_x = blockDim.x * blockIdx.x + threadIdx.x;

    arg0 += idx_x * 1*1 + idx_y * 1*1 * dims_KerComputeMacros[0][0];
    arg1 += idx_x * 1*1 + idx_y * 1*1 * dims_KerComputeMacros[1][0];
    arg2 += idx_x * 1*1 + idx_y * 1*1 * dims_KerComputeMacros[2][0];
    arg3 += idx_x * 1*9 + idx_y * 1*9 * dims_KerComputeMacros[3][0];

    if (idx_x < size0 && idx_y < size1) {
        ACC<double> argp0(dims_KerComputeMacros[0][0], arg0);
        ACC<double> argp1(dims_KerComputeMacros[1][0], arg1);
        ACC<double> argp2(dims_KerComputeMacros[2][0], arg2);
        const ACC<double> argp3(9, dims_KerComputeMacros[3][0], 0, arg3);
        KerComputeMacros_gpu(
         argp0, argp1, argp2, argp3, arg4,
         arg5, &arg6);
    }

} //End of cuda kernel wrapper function

//  ==================
//  Host stub function
//  ==================
#ifndef OPS_LAZY
void ops_par_loop_KerComputeMacros(char const *name, ops_block block, int dim, int* range,
 ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
 ops_arg arg4, ops_arg arg5, ops_arg arg6) {
#else
void ops_par_loop_KerComputeMacros_execute(ops_kernel_descriptor *desc) {
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
    if (!ops_checkpointing_before(args, 7, range, 11)) return;
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timing_realloc(block->instance, 11, "KerComputeMacros");
        block->instance->OPS_kernels[11].count++;
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

    if (xdim0 != dims_KerComputeMacros_h[0][0] || xdim1 != dims_KerComputeMacros_h[1][0] || xdim2 != dims_KerComputeMacros_h[2][0] || xdim3 != dims_KerComputeMacros_h[3][0]) {
        dims_KerComputeMacros_h[0][0] = xdim0;
        dims_KerComputeMacros_h[1][0] = xdim1;
        dims_KerComputeMacros_h[2][0] = xdim2;
        dims_KerComputeMacros_h[3][0] = xdim3;

        cutilSafeCall(block->instance->ostream(), cudaMemcpyToSymbol( dims_KerComputeMacros, dims_KerComputeMacros_h, sizeof(dims_KerComputeMacros)));
    }

    int *arg4h = (int *)arg4.data;
    int *arg5h = (int *)arg5.data;

    int x_size = MAX(0,end_indx[0]-start_indx[0]);
    int y_size = MAX(0,end_indx[1]-start_indx[1]);

    dim3 grid( (x_size-1)/block->instance->OPS_block_size_x+ 1, (y_size-1)/block->instance->OPS_block_size_y + 1, 1);
    dim3 tblock(block->instance->OPS_block_size_x,block->instance->OPS_block_size_y,block->instance->OPS_block_size_z);

    int consts_bytes = 0;

    consts_bytes += ROUND_UP(arg4.dim*sizeof(int));
    consts_bytes += ROUND_UP(arg5.dim*sizeof(int));

    reallocConstArrays(block->instance,consts_bytes);
    consts_bytes = 0;
    arg4.data = block->instance->OPS_consts_h + consts_bytes;
    arg4.data_d = block->instance->OPS_consts_d + consts_bytes;
    for (int d=0; d<arg4.dim; d++) ((int *)arg4.data)[d] = arg4h[d];
    consts_bytes += ROUND_UP(arg4.dim*sizeof(int));
    arg5.data = block->instance->OPS_consts_h + consts_bytes;
    arg5.data_d = block->instance->OPS_consts_d + consts_bytes;
    for (int d=0; d<arg5.dim; d++) ((int *)arg5.data)[d] = arg5h[d];
    consts_bytes += ROUND_UP(arg5.dim*sizeof(int));
    mvConstArraysToDevice(block->instance,consts_bytes);

    long long int dat0 = (block->instance->OPS_soa ? args[0].dat->type_size : args[0].dat->elem_size);
    long long int dat1 = (block->instance->OPS_soa ? args[1].dat->type_size : args[1].dat->elem_size);
    long long int dat2 = (block->instance->OPS_soa ? args[2].dat->type_size : args[2].dat->elem_size);
    long long int dat3 = (block->instance->OPS_soa ? args[3].dat->type_size : args[3].dat->elem_size);

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

//  =============
//  Halo exchange
//  =============
#ifndef OPS_LAZY
    ops_H_D_exchanges_device(args, 7);
    ops_halo_exchanges(args, 7, range);
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[11].mpi_time += __t2 - __t1;
    }


//call kernel wrapper function, passing in pointers to data
    if (x_size > 0 && y_size > 0) {

        ops_KerComputeMacros<<<grid, tblock >>> ( 
         (double *)p_a[0], (double *)p_a[1],
         (double *)p_a[2], (double *)p_a[3],
         (int *)arg4.data_d, (int *)arg5.data_d,
         *(double *)arg6.data,
        x_size, y_size);

    }

    cutilSafeCall(block->instance->ostream(), cudaGetLastError());

    if (block->instance->OPS_diags>1) {
        cutilSafeCall(block->instance->ostream(), cudaDeviceSynchronize());
        ops_timers_core(&__c1, &__t1);
        block->instance->OPS_kernels[11].time += __t1 - __t2;
    }

#ifndef OPS_LAZY
    ops_set_dirtybit_device(args, 7);
    ops_set_halo_dirtybit3(&args[0], range);
    ops_set_halo_dirtybit3(&args[1], range);
    ops_set_halo_dirtybit3(&args[2], range);
#endif

    if (block->instance->OPS_diags > 1) {
        //  ====================
        //  Update kernel record
        //  ====================
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[11].mpi_time += __t2 - __t1;
        block->instance->OPS_kernels[11].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg0);
        block->instance->OPS_kernels[11].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg1);
        block->instance->OPS_kernels[11].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg2);
        block->instance->OPS_kernels[11].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg3);
    }
}

#ifdef OPS_LAZY
void ops_par_loop_KerComputeMacros(char const *name, ops_block block, int dim, int* range,
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

    create_kerneldesc_and_enque("KerComputeMacros", args, 7, 11, dim, 1, range, block, ops_par_loop_KerComputeMacros_execute);
}
#endif
