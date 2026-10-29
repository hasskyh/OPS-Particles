// Auto-generated at 2026-10-29 00:14:13.148653 by ops-translator legacy


//  ==================
//  Host stub function
//  ==================
#ifndef OPS_LAZY
void ops_par_loop_KerComputeMacros(
    const char *name,
    ops_block block,
    int dim,
    int* range,
    ops_arg arg0, 
    ops_arg arg1, 
    ops_arg arg2, 
    ops_arg arg3, 
    ops_arg arg4, 
    ops_arg arg5, 
    ops_arg arg6
)
{
#else
void ops_par_loop_KerComputeMacros_execute(ops_kernel_descriptor *desc)
{
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

#if defined(CHECKPOINTING) && !defined(OPS_LAZY)
    if (!ops_checkpointing_before(args, 7, range, 11)) return;
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timing_realloc(block->instance, 11, "KerComputeMacros");
        block->instance->OPS_kernels[11].count++;
        ops_timers_core(&__c1, &__t1);
    }

#ifdef OPS_DEBUG
    ops_register_args(block->instance, args, "KerComputeMacros");
#endif


//  =================================================
//  compute locally allocated range for the sub-block
//  =================================================
    int start[2];
    int end[2];
#if defined(OPS_MPI) && !defined(OPS_LAZY)
    int arg_idx[2];
#endif

#if defined(OPS_LAZY) || !defined(OPS_MPI)
    for (int n = 0; n < 2; n++) {
        start[n] = range[2*n];
        end[n]   = range[2*n+1];
    }
#else
    if (compute_ranges(args, 7, block, range, start, end, arg_idx) < 0) return;
#endif

    int start0 = start[0];
    int end0 = end[0];
    int start1 = start[1];
    int end1 = end[1];

    //  =====================================================
    //  Initialize global variable with the dimension of dats
    //  =====================================================
    int xdim0_KerComputeMacros = args[0].dat->size[0];
    int xdim1_KerComputeMacros = args[1].dat->size[0];
    int xdim2_KerComputeMacros = args[2].dat->size[0];
    int xdim3_KerComputeMacros = args[3].dat->size[0];
    int ydim3_KerComputeMacros = args[3].dat->size[1];

//  =======================================================
//  Set up initial pointers and exchange halos if necessary
//  =======================================================
    int base0 = args[0].dat->base_offset;
    double * __restrict__ rho_p = (double *)(args[0].data_d + base0);

    int base1 = args[1].dat->base_offset;
    double * __restrict__ u_p = (double *)(args[1].data_d + base1);

    int base2 = args[2].dat->base_offset;
    double * __restrict__ v_p = (double *)(args[2].data_d + base2);

    int base3 = args[3].dat->base_offset;
    double * __restrict__ f_p = (double *)(args[3].data_d + base3);





    int consts_bytes = 0;

    int *arg4h = (int *)args[4].data;
    consts_bytes += ROUND_UP(args[4].dim*sizeof(int));
    int *arg5h = (int *)args[5].data;
    consts_bytes += ROUND_UP(args[5].dim*sizeof(int));
    double *arg6h = (double *)args[6].data;
    consts_bytes += ROUND_UP(args[6].dim*sizeof(double));

    reallocConstArrays(block->instance,consts_bytes);
    consts_bytes = 0;

    args[4].data = block->instance->OPS_consts_h + consts_bytes;
    args[4].data_d = block->instance->OPS_consts_d + consts_bytes;
    for (int d = 0; d < args[4].dim; d++) {
        ((int *)args[4].data)[d] = arg4h[d];
    }
    consts_bytes += ROUND_UP(args[4].dim*sizeof(int));

    args[5].data = block->instance->OPS_consts_h + consts_bytes;
    args[5].data_d = block->instance->OPS_consts_d + consts_bytes;
    for (int d = 0; d < args[5].dim; d++) {
        ((int *)args[5].data)[d] = arg5h[d];
    }
    consts_bytes += ROUND_UP(args[5].dim*sizeof(int));

    args[6].data = block->instance->OPS_consts_h + consts_bytes;
    args[6].data_d = block->instance->OPS_consts_d + consts_bytes;
    for (int d = 0; d < args[6].dim; d++) {
        ((double *)args[6].data)[d] = arg6h[d];
    }
    consts_bytes += ROUND_UP(args[6].dim*sizeof(double));

    mvConstArraysToDevice(block->instance,consts_bytes);

    int * __restrict__ cx = (int *)args[4].data_d;
    int * __restrict__ cy = (int *)args[5].data_d;
    double * __restrict__ cs = (double *)args[6].data_d;

#ifndef OPS_LAZY
//  ==============
//  Halo Exchanges
//  ==============
    ops_H_D_exchanges_device(args, 7);
    ops_halo_exchanges(args, 7,range);
    ops_H_D_exchanges_device(args, 7);
#endif //OPS_LAZY

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[11].mpi_time += __t2 - __t1;
    }

    #pragma omp target teams distribute parallel for collapse(2)
    for (int n_y = start1; n_y < end1; n_y++) {
        for (int n_x = start0; n_x < end0; n_x++) {

            ACC<double> rho(xdim0_KerComputeMacros, rho_p + n_x*1 + n_y * xdim0_KerComputeMacros*1);

            ACC<double> u(xdim1_KerComputeMacros, u_p + n_x*1 + n_y * xdim1_KerComputeMacros*1);

            ACC<double> v(xdim2_KerComputeMacros, v_p + n_x*1 + n_y * xdim2_KerComputeMacros*1);

#ifdef OPS_SOA
            const ACC<double> f(9, xdim3_KerComputeMacros, ydim3_KerComputeMacros, f_p + n_x*1 + n_y * xdim3_KerComputeMacros*1);
#else
            const ACC<double> f(9, xdim3_KerComputeMacros, ydim3_KerComputeMacros, f_p + 9*(n_x*1 + n_y * xdim3_KerComputeMacros*1));
#endif


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
}

if (block->instance->OPS_diags > 1) {
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
        block->instance->OPS_kernels[11].transfer += ops_compute_transfer(dim, start, end, &arg0);
        block->instance->OPS_kernels[11].transfer += ops_compute_transfer(dim, start, end, &arg1);
        block->instance->OPS_kernels[11].transfer += ops_compute_transfer(dim, start, end, &arg2);
        block->instance->OPS_kernels[11].transfer += ops_compute_transfer(dim, start, end, &arg3);
    }
}

#ifdef OPS_LAZY
void ops_par_loop_KerComputeMacros(
    const char *name,
    ops_block block,
    int dim,
    int* range,
    ops_arg arg0,
    ops_arg arg1,
    ops_arg arg2,
    ops_arg arg3,
    ops_arg arg4,
    ops_arg arg5,
    ops_arg arg6
)
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
