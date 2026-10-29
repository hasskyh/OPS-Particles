// Auto-generated at 2026-10-29 00:14:13.280554 by ops-translator legacy

__constant__ int dims_KerCollision [10][1];
static int dims_KerCollision_h [10][1] = {{0}};

//  =============
//  User function
//  =============
__device__ void KerCollision_gpu(ACC<double>& f_work,
  const ACC<double>& f,
  const ACC<double>& rho,
  const ACC<double>& u,
  const ACC<double>& v,
  const double *omega,
  const int *cx,
  const int *cy,
  const double *wi,
  const double *cs) {


  double rhoF = rho(0, 0);
  double uF = u(0, 0);
  double vF = v(0, 0);
  for (int i = 0; i < 9; i++) {
    double feq = calculate_feq2D(rhoF, uF, vF, cx[i], cy[i], wi[i], *cs);
    f_work(i, 0, 0) = (1. - (*omega)) * f(i, 0, 0) + (*omega) * feq;
  }

}


//  ============================
//  Cuda kernel wrapper function
//  ============================
__global__ void ops_KerCollision(
double* __restrict arg0,
double* __restrict arg1,
double* __restrict arg2,
double* __restrict arg3,
double* __restrict arg4,
const double arg5,
const int* __restrict arg6,
const int* __restrict arg7,
const double* __restrict arg8,
const double arg9,
int size0,
int size1) {
    int idx_y = blockDim.y * blockIdx.y + threadIdx.y;
    int idx_x = blockDim.x * blockIdx.x + threadIdx.x;

    arg0 += idx_x * 1*9 + idx_y * 1*9 * dims_KerCollision[0][0];
    arg1 += idx_x * 1*9 + idx_y * 1*9 * dims_KerCollision[1][0];
    arg2 += idx_x * 1*1 + idx_y * 1*1 * dims_KerCollision[2][0];
    arg3 += idx_x * 1*1 + idx_y * 1*1 * dims_KerCollision[3][0];
    arg4 += idx_x * 1*1 + idx_y * 1*1 * dims_KerCollision[4][0];

    if (idx_x < size0 && idx_y < size1) {
        ACC<double> argp0(9, dims_KerCollision[0][0], 0, arg0);
        const ACC<double> argp1(9, dims_KerCollision[1][0], 0, arg1);
        const ACC<double> argp2(dims_KerCollision[2][0], arg2);
        const ACC<double> argp3(dims_KerCollision[3][0], arg3);
        const ACC<double> argp4(dims_KerCollision[4][0], arg4);
        KerCollision_gpu(
         argp0, argp1, argp2, argp3, argp4,
         &arg5, arg6, arg7, arg8, &arg9);
    }

} //End of cuda kernel wrapper function

//  ==================
//  Host stub function
//  ==================
#ifndef OPS_LAZY
void ops_par_loop_KerCollision(char const *name, ops_block block, int dim, int* range,
 ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
 ops_arg arg4, ops_arg arg5, ops_arg arg6, ops_arg arg7,
 ops_arg arg8, ops_arg arg9) {
#else
void ops_par_loop_KerCollision_execute(ops_kernel_descriptor *desc) {
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
#endif

//  ======
//  Timing
//  ======
    double __t1, __t2, __c1, __c2;

    ops_arg args[10];

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

#if CHECKPOINTING && !OPS_LAZY
    if (!ops_checkpointing_before(args, 10, range, 12)) return;
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timing_realloc(block->instance, 12, "KerCollision");
        block->instance->OPS_kernels[12].count++;
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
    if (compute_ranges(args, 10, block, range, start_indx, end_indx, arg_idx) < 0) return;
#endif

    int xdim0 = args[0].dat->size[0];
    int xdim1 = args[1].dat->size[0];
    int xdim2 = args[2].dat->size[0];
    int xdim3 = args[3].dat->size[0];
    int xdim4 = args[4].dat->size[0];

    if (xdim0 != dims_KerCollision_h[0][0] || xdim1 != dims_KerCollision_h[1][0] || xdim2 != dims_KerCollision_h[2][0] || xdim3 != dims_KerCollision_h[3][0] || xdim4 != dims_KerCollision_h[4][0]) {
        dims_KerCollision_h[0][0] = xdim0;
        dims_KerCollision_h[1][0] = xdim1;
        dims_KerCollision_h[2][0] = xdim2;
        dims_KerCollision_h[3][0] = xdim3;
        dims_KerCollision_h[4][0] = xdim4;

        hipSafeCall(block->instance->ostream(), hipMemcpyToSymbol( dims_KerCollision, dims_KerCollision_h, sizeof(dims_KerCollision)));
    }

    int *arg6h = (int *)arg6.data;
    int *arg7h = (int *)arg7.data;
    double *arg8h = (double *)arg8.data;

    int x_size = MAX(0,end_indx[0]-start_indx[0]);
    int y_size = MAX(0,end_indx[1]-start_indx[1]);

    dim3 grid( (x_size-1)/block->instance->OPS_block_size_x+ 1, (y_size-1)/block->instance->OPS_block_size_y + 1, 1);
    dim3 tblock(block->instance->OPS_block_size_x,block->instance->OPS_block_size_y,block->instance->OPS_block_size_z);

    int consts_bytes = 0;

    consts_bytes += ROUND_UP(arg6.dim*sizeof(int));
    consts_bytes += ROUND_UP(arg7.dim*sizeof(int));
    consts_bytes += ROUND_UP(arg8.dim*sizeof(double));

    reallocConstArrays(block->instance,consts_bytes);
    consts_bytes = 0;
    arg6.data = block->instance->OPS_consts_h + consts_bytes;
    arg6.data_d = block->instance->OPS_consts_d + consts_bytes;
    for (int d=0; d<arg6.dim; d++) ((int *)arg6.data)[d] = arg6h[d];
    consts_bytes += ROUND_UP(arg6.dim*sizeof(int));
    arg7.data = block->instance->OPS_consts_h + consts_bytes;
    arg7.data_d = block->instance->OPS_consts_d + consts_bytes;
    for (int d=0; d<arg7.dim; d++) ((int *)arg7.data)[d] = arg7h[d];
    consts_bytes += ROUND_UP(arg7.dim*sizeof(int));
    arg8.data = block->instance->OPS_consts_h + consts_bytes;
    arg8.data_d = block->instance->OPS_consts_d + consts_bytes;
    for (int d=0; d<arg8.dim; d++) ((double *)arg8.data)[d] = arg8h[d];
    consts_bytes += ROUND_UP(arg8.dim*sizeof(double));
    mvConstArraysToDevice(block->instance,consts_bytes);

    long long int dat0 = (block->instance->OPS_soa ? args[0].dat->type_size : args[0].dat->elem_size);
    long long int dat1 = (block->instance->OPS_soa ? args[1].dat->type_size : args[1].dat->elem_size);
    long long int dat2 = (block->instance->OPS_soa ? args[2].dat->type_size : args[2].dat->elem_size);
    long long int dat3 = (block->instance->OPS_soa ? args[3].dat->type_size : args[3].dat->elem_size);
    long long int dat4 = (block->instance->OPS_soa ? args[4].dat->type_size : args[4].dat->elem_size);

    char *p_a[10];

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

//  =============
//  Halo exchange
//  =============
#ifndef OPS_LAZY
    ops_H_D_exchanges_device(args, 10);
    ops_halo_exchanges(args, 10, range);
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[12].mpi_time += __t2 - __t1;
    }


//call kernel wrapper function, passing in pointers to data
    if (x_size > 0 && y_size > 0) {

        ops_KerCollision<<<grid, tblock >>> ( 
         (double *)p_a[0], (double *)p_a[1],
         (double *)p_a[2], (double *)p_a[3],
         (double *)p_a[4], *(double *)arg5.data,
         (int *)arg6.data_d, (int *)arg7.data_d,
         (double *)arg8.data_d, *(double *)arg9.data,
        x_size, y_size);

    }

    hipSafeCall(block->instance->ostream(), hipGetLastError());

    if (block->instance->OPS_diags>1) {
        hipSafeCall(block->instance->ostream(), hipDeviceSynchronize());
        ops_timers_core(&__c1, &__t1);
        block->instance->OPS_kernels[12].time += __t1 - __t2;
    }

#ifndef OPS_LAZY
    ops_set_dirtybit_device(args, 10);
    ops_set_halo_dirtybit3(&args[0], range);
#endif

    if (block->instance->OPS_diags > 1) {
        //  ====================
        //  Update kernel record
        //  ====================
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[12].mpi_time += __t2 - __t1;
        block->instance->OPS_kernels[12].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg0);
        block->instance->OPS_kernels[12].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg1);
        block->instance->OPS_kernels[12].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg2);
        block->instance->OPS_kernels[12].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg3);
        block->instance->OPS_kernels[12].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg4);
    }
}

#ifdef OPS_LAZY
void ops_par_loop_KerCollision(char const *name, ops_block block, int dim, int* range,
 ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
 ops_arg arg4, ops_arg arg5, ops_arg arg6, ops_arg arg7,
 ops_arg arg8, ops_arg arg9)
{
    ops_arg args[10];

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

    create_kerneldesc_and_enque("KerCollision", args, 10, 12, dim, 1, range, block, ops_par_loop_KerCollision_execute);
}
#endif
