// Auto-generated at 2026-10-29 00:13:49.395713 by ops-translator legacy

__constant__ int dims_advec_mom_kernel2_y [4][1];
static int dims_advec_mom_kernel2_y_h [4][1] = {{0}};

//  =============
//  User function
//  =============
__device__ inline void advec_mom_kernel2_y_gpu(ACC<double> &vel1,
  const ACC<double> &node_mass_post,
  const ACC<double> &node_mass_pre,
  const ACC<double> &mom_flux) {


  vel1(0,0) = ( vel1(0,0) * node_mass_pre(0,0)  +
    mom_flux(0,-1) - mom_flux(0,0) ) / node_mass_post(0,0);
}


//  ============================
//  Cuda kernel wrapper function
//  ============================
__global__ void ops_advec_mom_kernel2_y(
double* __restrict arg0,
double* __restrict arg1,
double* __restrict arg2,
double* __restrict arg3,
int size0,
int size1) {
    int idx_y = blockDim.y * blockIdx.y + threadIdx.y;
    int idx_x = blockDim.x * blockIdx.x + threadIdx.x;

    arg0 += idx_x * 1*1 + idx_y * 1*1 * dims_advec_mom_kernel2_y[0][0];
    arg1 += idx_x * 1*1 + idx_y * 1*1 * dims_advec_mom_kernel2_y[1][0];
    arg2 += idx_x * 1*1 + idx_y * 1*1 * dims_advec_mom_kernel2_y[2][0];
    arg3 += idx_x * 1*1 + idx_y * 1*1 * dims_advec_mom_kernel2_y[3][0];

    if (idx_x < size0 && idx_y < size1) {
        ACC<double> argp0(dims_advec_mom_kernel2_y[0][0], arg0);
        const ACC<double> argp1(dims_advec_mom_kernel2_y[1][0], arg1);
        const ACC<double> argp2(dims_advec_mom_kernel2_y[2][0], arg2);
        const ACC<double> argp3(dims_advec_mom_kernel2_y[3][0], arg3);
        advec_mom_kernel2_y_gpu(
         argp0, argp1, argp2, argp3);
    }

} //End of cuda kernel wrapper function

//  ==================
//  Host stub function
//  ==================
#ifndef OPS_LAZY
void ops_par_loop_advec_mom_kernel2_y(char const *name, ops_block block, int dim, int* range,
 ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3) {
#else
void ops_par_loop_advec_mom_kernel2_y_execute(ops_kernel_descriptor *desc) {
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
    if (!ops_checkpointing_before(args, 4, range, 22)) return;
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timing_realloc(block->instance, 22, "advec_mom_kernel2_y");
        block->instance->OPS_kernels[22].count++;
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
    int xdim2 = args[2].dat->size[0];
    int xdim3 = args[3].dat->size[0];

    if (xdim0 != dims_advec_mom_kernel2_y_h[0][0] || xdim1 != dims_advec_mom_kernel2_y_h[1][0] || xdim2 != dims_advec_mom_kernel2_y_h[2][0] || xdim3 != dims_advec_mom_kernel2_y_h[3][0]) {
        dims_advec_mom_kernel2_y_h[0][0] = xdim0;
        dims_advec_mom_kernel2_y_h[1][0] = xdim1;
        dims_advec_mom_kernel2_y_h[2][0] = xdim2;
        dims_advec_mom_kernel2_y_h[3][0] = xdim3;

        cutilSafeCall(block->instance->ostream(), cudaMemcpyToSymbol( dims_advec_mom_kernel2_y, dims_advec_mom_kernel2_y_h, sizeof(dims_advec_mom_kernel2_y)));
    }


    int x_size = MAX(0,end_indx[0]-start_indx[0]);
    int y_size = MAX(0,end_indx[1]-start_indx[1]);

    dim3 grid( (x_size-1)/block->instance->OPS_block_size_x+ 1, (y_size-1)/block->instance->OPS_block_size_y + 1, 1);
    dim3 tblock(block->instance->OPS_block_size_x,block->instance->OPS_block_size_y,block->instance->OPS_block_size_z);



    long long int dat0 = (block->instance->OPS_soa ? args[0].dat->type_size : args[0].dat->elem_size);
    long long int dat1 = (block->instance->OPS_soa ? args[1].dat->type_size : args[1].dat->elem_size);
    long long int dat2 = (block->instance->OPS_soa ? args[2].dat->type_size : args[2].dat->elem_size);
    long long int dat3 = (block->instance->OPS_soa ? args[3].dat->type_size : args[3].dat->elem_size);

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
    ops_H_D_exchanges_device(args, 4);
    ops_halo_exchanges(args, 4, range);
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[22].mpi_time += __t2 - __t1;
    }


//call kernel wrapper function, passing in pointers to data
    if (x_size > 0 && y_size > 0) {

        ops_advec_mom_kernel2_y<<<grid, tblock >>> ( 
         (double *)p_a[0], (double *)p_a[1],
         (double *)p_a[2], (double *)p_a[3],
        x_size, y_size);

    }

    cutilSafeCall(block->instance->ostream(), cudaGetLastError());

    if (block->instance->OPS_diags>1) {
        cutilSafeCall(block->instance->ostream(), cudaDeviceSynchronize());
        ops_timers_core(&__c1, &__t1);
        block->instance->OPS_kernels[22].time += __t1 - __t2;
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
        block->instance->OPS_kernels[22].mpi_time += __t2 - __t1;
        block->instance->OPS_kernels[22].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg0);
        block->instance->OPS_kernels[22].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg1);
        block->instance->OPS_kernels[22].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg2);
        block->instance->OPS_kernels[22].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg3);
    }
}

#ifdef OPS_LAZY
void ops_par_loop_advec_mom_kernel2_y(char const *name, ops_block block, int dim, int* range,
 ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3)
{
    ops_arg args[4];

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;
    args[3] = arg3;

    create_kerneldesc_and_enque("advec_mom_kernel2_y", args, 4, 22, dim, 1, range, block, ops_par_loop_advec_mom_kernel2_y_execute);
}
#endif
