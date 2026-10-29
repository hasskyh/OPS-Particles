// Auto-generated at 2026-10-29 00:14:11.429328 by ops-translator legacy

__constant__ int dims_residue_eval [6][1];
static int dims_residue_eval_h [6][1] = {{0}};

//  =============
//  User function
//  =============
__device__ void residue_eval_gpu(const ACC<double> &der1,
  const ACC<double> &der2,
  const ACC<double> &der3,
  ACC<double> &rho_res,
  ACC<double> &rhou_res,
  ACC<double> &rhoE_res) {
  rho_res(0) = der1(0);
  rhou_res(0) = der2(0);
  rhoE_res(0) = der3(0);
  }


//  ============================
//  Cuda kernel wrapper function
//  ============================
__global__ void ops_residue_eval(
double* __restrict arg0,
double* __restrict arg1,
double* __restrict arg2,
double* __restrict arg3,
double* __restrict arg4,
double* __restrict arg5,
int size0) {
    int idx_x = blockDim.x * blockIdx.x + threadIdx.x;

    arg0 += idx_x * 1*1;
    arg1 += idx_x * 1*1;
    arg2 += idx_x * 1*1;
    arg3 += idx_x * 1*1;
    arg4 += idx_x * 1*1;
    arg5 += idx_x * 1*1;

    if (idx_x < size0) {
        const ACC<double> argp0(arg0);
        const ACC<double> argp1(arg1);
        const ACC<double> argp2(arg2);
        ACC<double> argp3(arg3);
        ACC<double> argp4(arg4);
        ACC<double> argp5(arg5);
        residue_eval_gpu(
         argp0, argp1, argp2, argp3, argp4,
         argp5);
    }

} //End of cuda kernel wrapper function

//  ==================
//  Host stub function
//  ==================
#ifndef OPS_LAZY
void ops_par_loop_residue_eval(char const *name, ops_block block, int dim, int* range,
 ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
 ops_arg arg4, ops_arg arg5) {
#else
void ops_par_loop_residue_eval_execute(ops_kernel_descriptor *desc) {
    ops_block block = desc->block;
    int dim = desc->dim;
    int *range = desc->range;
    ops_arg arg0 = desc->args[0];
    ops_arg arg1 = desc->args[1];
    ops_arg arg2 = desc->args[2];
    ops_arg arg3 = desc->args[3];
    ops_arg arg4 = desc->args[4];
    ops_arg arg5 = desc->args[5];
#endif

//  ======
//  Timing
//  ======
    double __t1, __t2, __c1, __c2;

    ops_arg args[6];

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;
    args[3] = arg3;
    args[4] = arg4;
    args[5] = arg5;

#if CHECKPOINTING && !OPS_LAZY
    if (!ops_checkpointing_before(args, 6, range, 5)) return;
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timing_realloc(block->instance, 5, "residue_eval");
        block->instance->OPS_kernels[5].count++;
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
    if (compute_ranges(args, 6, block, range, start_indx, end_indx, arg_idx) < 0) return;
#endif

    int xdim0 = args[0].dat->size[0];
    int xdim1 = args[1].dat->size[0];
    int xdim2 = args[2].dat->size[0];
    int xdim3 = args[3].dat->size[0];
    int xdim4 = args[4].dat->size[0];
    int xdim5 = args[5].dat->size[0];

    if (xdim0 != dims_residue_eval_h[0][0] || xdim1 != dims_residue_eval_h[1][0] || xdim2 != dims_residue_eval_h[2][0] || xdim3 != dims_residue_eval_h[3][0] || xdim4 != dims_residue_eval_h[4][0] || xdim5 != dims_residue_eval_h[5][0]) {
        dims_residue_eval_h[0][0] = xdim0;
        dims_residue_eval_h[1][0] = xdim1;
        dims_residue_eval_h[2][0] = xdim2;
        dims_residue_eval_h[3][0] = xdim3;
        dims_residue_eval_h[4][0] = xdim4;
        dims_residue_eval_h[5][0] = xdim5;

        cutilSafeCall(block->instance->ostream(), cudaMemcpyToSymbol( dims_residue_eval, dims_residue_eval_h, sizeof(dims_residue_eval)));
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

    char *p_a[6];

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

//  =============
//  Halo exchange
//  =============
#ifndef OPS_LAZY
    ops_H_D_exchanges_device(args, 6);
    ops_halo_exchanges(args, 6, range);
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[5].mpi_time += __t2 - __t1;
    }


//call kernel wrapper function, passing in pointers to data
    if (x_size > 0) {

        ops_residue_eval<<<grid, tblock >>> ( 
         (double *)p_a[0], (double *)p_a[1],
         (double *)p_a[2], (double *)p_a[3],
         (double *)p_a[4], (double *)p_a[5],
        x_size);

    }

    cutilSafeCall(block->instance->ostream(), cudaGetLastError());

    if (block->instance->OPS_diags>1) {
        cutilSafeCall(block->instance->ostream(), cudaDeviceSynchronize());
        ops_timers_core(&__c1, &__t1);
        block->instance->OPS_kernels[5].time += __t1 - __t2;
    }

#ifndef OPS_LAZY
    ops_set_dirtybit_device(args, 6);
    ops_set_halo_dirtybit3(&args[3], range);
    ops_set_halo_dirtybit3(&args[4], range);
    ops_set_halo_dirtybit3(&args[5], range);
#endif

    if (block->instance->OPS_diags > 1) {
        //  ====================
        //  Update kernel record
        //  ====================
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[5].mpi_time += __t2 - __t1;
        block->instance->OPS_kernels[5].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg0);
        block->instance->OPS_kernels[5].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg1);
        block->instance->OPS_kernels[5].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg2);
        block->instance->OPS_kernels[5].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg3);
        block->instance->OPS_kernels[5].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg4);
        block->instance->OPS_kernels[5].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg5);
    }
}

#ifdef OPS_LAZY
void ops_par_loop_residue_eval(char const *name, ops_block block, int dim, int* range,
 ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
 ops_arg arg4, ops_arg arg5)
{
    ops_arg args[6];

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;
    args[3] = arg3;
    args[4] = arg4;
    args[5] = arg5;

    create_kerneldesc_and_enque("residue_eval", args, 6, 5, dim, 1, range, block, ops_par_loop_residue_eval_execute);
}
#endif
