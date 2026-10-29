// Auto-generated at 2026-10-29 00:13:50.539934 by ops-translator legacy

__constant__ int dims_initialise_chunk_kernel_yy [2][1];
static int dims_initialise_chunk_kernel_yy_h [2][1] = {{0}};

//  =============
//  User function
//  =============
__device__ void initialise_chunk_kernel_yy_gpu(ACC<int> &yy,
  int *idx) {
  yy(0,0) = idx[1]-2;
}


//  ============================
//  Cuda kernel wrapper function
//  ============================
__global__ void ops_initialise_chunk_kernel_yy(
int* __restrict arg0,
int arg_idx0, int arg_idx1,
int size0,
int size1) {
    int idx_y = blockDim.y * blockIdx.y + threadIdx.y;
    int idx_x = blockDim.x * blockIdx.x + threadIdx.x;

    int arg_idx[2];
    arg_idx[0] = arg_idx0+idx_x;
    arg_idx[1] = arg_idx1+idx_y;

    arg0 += idx_x * 0*1 + idx_y * 1*1 * dims_initialise_chunk_kernel_yy[0][0];

    if (idx_x < size0 && idx_y < size1) {
        ACC<int> argp0(dims_initialise_chunk_kernel_yy[0][0], arg0);
        initialise_chunk_kernel_yy_gpu(
         argp0, arg_idx);
    }

} //End of cuda kernel wrapper function

//  ==================
//  Host stub function
//  ==================
#ifndef OPS_LAZY
void ops_par_loop_initialise_chunk_kernel_yy(char const *name, ops_block block, int dim, int* range,
 ops_arg arg0, ops_arg arg1) {
#else
void ops_par_loop_initialise_chunk_kernel_yy_execute(ops_kernel_descriptor *desc) {
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
    if (!ops_checkpointing_before(args, 2, range, 33)) return;
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timing_realloc(block->instance, 33, "initialise_chunk_kernel_yy");
        block->instance->OPS_kernels[33].count++;
        ops_timers_core(&__c1, &__t1);
    }

//  =================================================
//  compute locally allocated range for the sub-block
//  =================================================
    int start_indx[2];
    int end_indx[2];
    int arg_idx[2];

#if defined(OPS_LAZY) || !defined(OPS_MPI)
    for (int n = 0; n < 2; n++) {
        start_indx[n] = range[2*n];
        end_indx[n]   = range[2*n+1];
    }
#else
    if (compute_ranges(args, 2, block, range, start_indx, end_indx, arg_idx) < 0) return;
#endif

#if defined(OPS_MPI)
#if defined(OPS_LAZY)
    sub_block_list sb = OPS_sub_block_list[block->index];
    arg_idx[0] = sb->decomp_disp[0]+start_indx[0];
    arg_idx[1] = sb->decomp_disp[1]+start_indx[1];
#endif //OPS_LAZY
#else //OPS_MPI
    arg_idx[0] = start_indx[0];
    arg_idx[1] = start_indx[1];
#endif //OPS_MPI

    int xdim0 = args[0].dat->size[0];

    if (xdim0 != dims_initialise_chunk_kernel_yy_h[0][0]) {
        dims_initialise_chunk_kernel_yy_h[0][0] = xdim0;

        hipSafeCall(block->instance->ostream(), hipMemcpyToSymbol( dims_initialise_chunk_kernel_yy, dims_initialise_chunk_kernel_yy_h, sizeof(dims_initialise_chunk_kernel_yy)));
    }


    int x_size = MAX(0,end_indx[0]-start_indx[0]);
    int y_size = MAX(0,end_indx[1]-start_indx[1]);

    dim3 grid( (x_size-1)/block->instance->OPS_block_size_x+ 1, (y_size-1)/block->instance->OPS_block_size_y + 1, 1);
    dim3 tblock(block->instance->OPS_block_size_x,block->instance->OPS_block_size_y,block->instance->OPS_block_size_z);



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
        block->instance->OPS_kernels[33].mpi_time += __t2 - __t1;
    }


//call kernel wrapper function, passing in pointers to data
    if (x_size > 0 && y_size > 0) {

        ops_initialise_chunk_kernel_yy<<<grid, tblock >>> ( 
         (int *)p_a[0], arg_idx[0], arg_idx[1],
        x_size, y_size);

    }

    hipSafeCall(block->instance->ostream(), hipGetLastError());

    if (block->instance->OPS_diags>1) {
        hipSafeCall(block->instance->ostream(), hipDeviceSynchronize());
        ops_timers_core(&__c1, &__t1);
        block->instance->OPS_kernels[33].time += __t1 - __t2;
    }

#ifndef OPS_LAZY
    ops_set_dirtybit_device(args, 2);
    ops_set_halo_dirtybit3(&args[0], range);
#endif

    if (block->instance->OPS_diags > 1) {
        //  ====================
        //  Update kernel record
        //  ====================
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[33].mpi_time += __t2 - __t1;
        block->instance->OPS_kernels[33].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg0);
    }
}

#ifdef OPS_LAZY
void ops_par_loop_initialise_chunk_kernel_yy(char const *name, ops_block block, int dim, int* range,
 ops_arg arg0, ops_arg arg1)
{
    ops_arg args[2];

    args[0] = arg0;
    args[1] = arg1;

    create_kerneldesc_and_enque("initialise_chunk_kernel_yy", args, 2, 33, dim, 1, range, block, ops_par_loop_initialise_chunk_kernel_yy_execute);
}
#endif
