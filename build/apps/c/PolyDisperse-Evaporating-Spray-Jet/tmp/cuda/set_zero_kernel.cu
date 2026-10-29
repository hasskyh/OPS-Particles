// Auto-generated at 2026-10-29 00:14:13.720718 by ops-translator legacy

__constant__ int dims_set_zero [1][2];
static int dims_set_zero_h [1][2] = {{0}};

//  =============
//  User function
//  =============
__device__ void set_zero_gpu(ACC<double> &A) {
  A(0,0,0) = 0.0;
}


//  ============================
//  Cuda kernel wrapper function
//  ============================
__global__ void ops_set_zero(
double* __restrict arg0,
int size0,
int size1,
int size2) {
    int idx_z = blockDim.z * blockIdx.z + threadIdx.z;
    int idx_y = blockDim.y * blockIdx.y + threadIdx.y;
    int idx_x = blockDim.x * blockIdx.x + threadIdx.x;

    arg0 += idx_x * 1*1 + idx_y * 1*1 * dims_set_zero[0][0] + idx_z * 1*1 * dims_set_zero[0][0] * dims_set_zero[0][1];

    if (idx_x < size0 && idx_y < size1 && idx_z < size2) {
        ACC<double> argp0(dims_set_zero[0][0], dims_set_zero[0][1], arg0);
        set_zero_gpu(
         argp0);
    }

} //End of cuda kernel wrapper function

//  ==================
//  Host stub function
//  ==================
#ifndef OPS_LAZY
void ops_par_loop_set_zero(char const *name, ops_block block, int dim, int* range,
 ops_arg arg0) {
#else
void ops_par_loop_set_zero_execute(ops_kernel_descriptor *desc) {
    ops_block block = desc->block;
    int dim = desc->dim;
    int *range = desc->range;
    ops_arg arg0 = desc->args[0];
#endif

//  ======
//  Timing
//  ======
    double __t1, __t2, __c1, __c2;

    ops_arg args[1];

    args[0] = arg0;

#if CHECKPOINTING && !OPS_LAZY
    if (!ops_checkpointing_before(args, 1, range, 0)) return;
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timing_realloc(block->instance, 0, "set_zero");
        block->instance->OPS_kernels[0].count++;
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
    if (compute_ranges(args, 1, block, range, start_indx, end_indx, arg_idx) < 0) return;
#endif

    int xdim0 = args[0].dat->size[0];
    int ydim0 = args[0].dat->size[1];

    if (xdim0 != dims_set_zero_h[0][0] || ydim0 != dims_set_zero_h[0][1]) {
        dims_set_zero_h[0][0] = xdim0;
        dims_set_zero_h[0][1] = ydim0;

        cutilSafeCall(block->instance->ostream(), cudaMemcpyToSymbol( dims_set_zero, dims_set_zero_h, sizeof(dims_set_zero)));
    }


    int x_size = MAX(0,end_indx[0]-start_indx[0]);
    int y_size = MAX(0,end_indx[1]-start_indx[1]);
    int z_size = MAX(0,end_indx[2]-start_indx[2]);

    dim3 grid( (x_size-1)/block->instance->OPS_block_size_x+ 1, (y_size-1)/block->instance->OPS_block_size_y + 1, (z_size-1)/block->instance->OPS_block_size_z +1);
    dim3 tblock(block->instance->OPS_block_size_x,block->instance->OPS_block_size_y,block->instance->OPS_block_size_z);



    long long int dat0 = (block->instance->OPS_soa ? args[0].dat->type_size : args[0].dat->elem_size);

    char *p_a[1];

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
    ops_H_D_exchanges_device(args, 1);
    ops_halo_exchanges(args, 1, range);
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[0].mpi_time += __t2 - __t1;
    }


//call kernel wrapper function, passing in pointers to data
    if (x_size > 0 && y_size > 0 && z_size > 0) {

        ops_set_zero<<<grid, tblock >>> ( 
         (double *)p_a[0],
        x_size, y_size, z_size);

    }

    cutilSafeCall(block->instance->ostream(), cudaGetLastError());

    if (block->instance->OPS_diags>1) {
        cutilSafeCall(block->instance->ostream(), cudaDeviceSynchronize());
        ops_timers_core(&__c1, &__t1);
        block->instance->OPS_kernels[0].time += __t1 - __t2;
    }

#ifndef OPS_LAZY
    ops_set_dirtybit_device(args, 1);
    ops_set_halo_dirtybit3(&args[0], range);
#endif

    if (block->instance->OPS_diags > 1) {
        //  ====================
        //  Update kernel record
        //  ====================
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[0].mpi_time += __t2 - __t1;
        block->instance->OPS_kernels[0].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg0);
    }
}

#ifdef OPS_LAZY
void ops_par_loop_set_zero(char const *name, ops_block block, int dim, int* range,
 ops_arg arg0)
{
    ops_arg args[1];

    args[0] = arg0;

    create_kerneldesc_and_enque("set_zero", args, 1, 0, dim, 1, range, block, ops_par_loop_set_zero_execute);
}
#endif
