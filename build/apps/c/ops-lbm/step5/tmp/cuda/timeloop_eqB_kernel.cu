// Auto-generated at 2026-10-29 00:14:12.352728 by ops-translator legacy

__constant__ int dims_timeloop_eqB [4][1];
static int dims_timeloop_eqB_h [4][1] = {{0}};

//  =============
//  User function
//  =============
__device__ void timeloop_eqB_gpu(ACC<double> &N,
  const ACC<double> &workArray,
  const int* cx,
  const int* cy) {

    for (int f = 1; f < 9; f++) {
        N(f,0,0) = workArray(f,-cx[f],-cy[f]);
    }
}


//  ============================
//  Cuda kernel wrapper function
//  ============================
__global__ void ops_timeloop_eqB(
double* __restrict arg0,
double* __restrict arg1,
const int* __restrict arg2,
const int* __restrict arg3,
int size0,
int size1) {
    int idx_y = blockDim.y * blockIdx.y + threadIdx.y;
    int idx_x = blockDim.x * blockIdx.x + threadIdx.x;

    arg0 += idx_x * 1*9 + idx_y * 1*9 * dims_timeloop_eqB[0][0];
    arg1 += idx_x * 1*9 + idx_y * 1*9 * dims_timeloop_eqB[1][0];

    if (idx_x < size0 && idx_y < size1) {
        ACC<double> argp0(9, dims_timeloop_eqB[0][0], 0, arg0);
        const ACC<double> argp1(9, dims_timeloop_eqB[1][0], 0, arg1);
        timeloop_eqB_gpu(
         argp0, argp1, arg2, arg3);
    }

} //End of cuda kernel wrapper function

//  ==================
//  Host stub function
//  ==================
#ifndef OPS_LAZY
void ops_par_loop_timeloop_eqB(char const *name, ops_block block, int dim, int* range,
 ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3) {
#else
void ops_par_loop_timeloop_eqB_execute(ops_kernel_descriptor *desc) {
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
    if (!ops_checkpointing_before(args, 4, range, 3)) return;
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timing_realloc(block->instance, 3, "timeloop_eqB");
        block->instance->OPS_kernels[3].count++;
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

    if (xdim0 != dims_timeloop_eqB_h[0][0] || xdim1 != dims_timeloop_eqB_h[1][0]) {
        dims_timeloop_eqB_h[0][0] = xdim0;
        dims_timeloop_eqB_h[1][0] = xdim1;

        cutilSafeCall(block->instance->ostream(), cudaMemcpyToSymbol( dims_timeloop_eqB, dims_timeloop_eqB_h, sizeof(dims_timeloop_eqB)));
    }

    int *arg2h = (int *)arg2.data;
    int *arg3h = (int *)arg3.data;

    int x_size = MAX(0,end_indx[0]-start_indx[0]);
    int y_size = MAX(0,end_indx[1]-start_indx[1]);

    dim3 grid( (x_size-1)/block->instance->OPS_block_size_x+ 1, (y_size-1)/block->instance->OPS_block_size_y + 1, 1);
    dim3 tblock(block->instance->OPS_block_size_x,block->instance->OPS_block_size_y,block->instance->OPS_block_size_z);

    int consts_bytes = 0;

    consts_bytes += ROUND_UP(arg2.dim*sizeof(int));
    consts_bytes += ROUND_UP(arg3.dim*sizeof(int));

    reallocConstArrays(block->instance,consts_bytes);
    consts_bytes = 0;
    arg2.data = block->instance->OPS_consts_h + consts_bytes;
    arg2.data_d = block->instance->OPS_consts_d + consts_bytes;
    for (int d=0; d<arg2.dim; d++) ((int *)arg2.data)[d] = arg2h[d];
    consts_bytes += ROUND_UP(arg2.dim*sizeof(int));
    arg3.data = block->instance->OPS_consts_h + consts_bytes;
    arg3.data_d = block->instance->OPS_consts_d + consts_bytes;
    for (int d=0; d<arg3.dim; d++) ((int *)arg3.data)[d] = arg3h[d];
    consts_bytes += ROUND_UP(arg3.dim*sizeof(int));
    mvConstArraysToDevice(block->instance,consts_bytes);

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
        block->instance->OPS_kernels[3].mpi_time += __t2 - __t1;
    }


//call kernel wrapper function, passing in pointers to data
    if (x_size > 0 && y_size > 0) {

        ops_timeloop_eqB<<<grid, tblock >>> ( 
         (double *)p_a[0], (double *)p_a[1],
         (int *)arg2.data_d, (int *)arg3.data_d,
        x_size, y_size);

    }

    cutilSafeCall(block->instance->ostream(), cudaGetLastError());

    if (block->instance->OPS_diags>1) {
        cutilSafeCall(block->instance->ostream(), cudaDeviceSynchronize());
        ops_timers_core(&__c1, &__t1);
        block->instance->OPS_kernels[3].time += __t1 - __t2;
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
        block->instance->OPS_kernels[3].mpi_time += __t2 - __t1;
        block->instance->OPS_kernels[3].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg0);
        block->instance->OPS_kernels[3].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg1);
    }
}

#ifdef OPS_LAZY
void ops_par_loop_timeloop_eqB(char const *name, ops_block block, int dim, int* range,
 ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3)
{
    ops_arg args[4];

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;
    args[3] = arg3;

    create_kerneldesc_and_enque("timeloop_eqB", args, 4, 3, dim, 1, range, block, ops_par_loop_timeloop_eqB_execute);
}
#endif
