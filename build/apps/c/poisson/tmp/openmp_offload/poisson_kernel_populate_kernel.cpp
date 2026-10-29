// Auto-generated at 2026-10-29 00:13:46.444061 by ops-translator legacy


//  ==================
//  Host stub function
//  ==================
#ifndef OPS_LAZY
void ops_par_loop_poisson_kernel_populate(
    const char *name,
    ops_block block,
    int dim,
    int* range,
    ops_arg arg0, 
    ops_arg arg1, 
    ops_arg arg2, 
    ops_arg arg3, 
    ops_arg arg4, 
    ops_arg arg5
)
{
#else
void ops_par_loop_poisson_kernel_populate_execute(ops_kernel_descriptor *desc)
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

#if defined(CHECKPOINTING) && !defined(OPS_LAZY)
    if (!ops_checkpointing_before(args, 6, range, 0)) return;
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timing_realloc(block->instance, 0, "poisson_kernel_populate");
        block->instance->OPS_kernels[0].count++;
        ops_timers_core(&__c1, &__t1);
    }

#ifdef OPS_DEBUG
    ops_register_args(block->instance, args, "poisson_kernel_populate");
#endif


//  =================================================
//  compute locally allocated range for the sub-block
//  =================================================
    int start[2];
    int end[2];
    int arg_idx[2];

#if defined(OPS_LAZY) || !defined(OPS_MPI)
    for (int n = 0; n < 2; n++) {
        start[n] = range[2*n];
        end[n]   = range[2*n+1];
    }
#else
    if (compute_ranges(args, 6, block, range, start, end, arg_idx) < 0) return;
#endif

    int start0 = start[0];
    int end0 = end[0];
    int start1 = start[1];
    int end1 = end[1];

#if defined(OPS_MPI)
#if defined(OPS_LAZY)
    sub_block_list sb = OPS_sub_block_list[block->index];
    arg_idx[0] = sb->decomp_disp[0];
    arg_idx[1] = sb->decomp_disp[1];
#else
    arg_idx[0] -= start[0];
    arg_idx[1] -= start[1];
#endif
#else //OPS_MPI
    arg_idx[0] = 0;
    arg_idx[1] = 0;
#endif //OPS_MPI

//  =====================================================
//  Initialize global variable with the dimension of dats
//  =====================================================
    int xdim3_poisson_kernel_populate = args[3].dat->size[0];
    int xdim4_poisson_kernel_populate = args[4].dat->size[0];
    int xdim5_poisson_kernel_populate = args[5].dat->size[0];

//  =======================================================
//  Set up initial pointers and exchange halos if necessary
//  =======================================================


    int base3 = args[3].dat->base_offset;
    double * __restrict__ u_p = (double *)(args[3].data_d + base3);

    int base4 = args[4].dat->base_offset;
    double * __restrict__ f_p = (double *)(args[4].data_d + base4);

    int base5 = args[5].dat->base_offset;
    double * __restrict__ ref_p = (double *)(args[5].data_d + base5);


    int consts_bytes = 0;

    int *arg0h = (int *)args[0].data;
    consts_bytes += ROUND_UP(args[0].dim*sizeof(int));
    int *arg1h = (int *)args[1].data;
    consts_bytes += ROUND_UP(args[1].dim*sizeof(int));

    reallocConstArrays(block->instance,consts_bytes);
    consts_bytes = 0;

    args[0].data = block->instance->OPS_consts_h + consts_bytes;
    args[0].data_d = block->instance->OPS_consts_d + consts_bytes;
    for (int d = 0; d < args[0].dim; d++) {
        ((int *)args[0].data)[d] = arg0h[d];
    }
    consts_bytes += ROUND_UP(args[0].dim*sizeof(int));

    args[1].data = block->instance->OPS_consts_h + consts_bytes;
    args[1].data_d = block->instance->OPS_consts_d + consts_bytes;
    for (int d = 0; d < args[1].dim; d++) {
        ((int *)args[1].data)[d] = arg1h[d];
    }
    consts_bytes += ROUND_UP(args[1].dim*sizeof(int));

    mvConstArraysToDevice(block->instance,consts_bytes);

    int * __restrict__ dispx = (int *)args[0].data_d;
    int * __restrict__ dispy = (int *)args[1].data_d;

#ifndef OPS_LAZY
//  ==============
//  Halo Exchanges
//  ==============
    ops_H_D_exchanges_device(args, 6);
    ops_halo_exchanges(args, 6,range);
    ops_H_D_exchanges_device(args, 6);
#endif //OPS_LAZY

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[0].mpi_time += __t2 - __t1;
    }

    #pragma omp target teams distribute parallel for collapse(2)
    for (int n_y = start1; n_y < end1; n_y++) {
        for (int n_x = start0; n_x < end0; n_x++) {

            int idx[] = {arg_idx[0]+n_x, arg_idx[1]+n_y};
            ACC<double> u(xdim3_poisson_kernel_populate, u_p + n_x*1 + n_y * xdim3_poisson_kernel_populate*1);

            ACC<double> f(xdim4_poisson_kernel_populate, f_p + n_x*1 + n_y * xdim4_poisson_kernel_populate*1);

            ACC<double> ref(xdim5_poisson_kernel_populate, ref_p + n_x*1 + n_y * xdim5_poisson_kernel_populate*1);

            
  double x = dx * (double)(idx[0]+dispx[0]);
  double y = dy * (double)(idx[1]+dispy[0]);

  u(0,0) = myfun(sin(M_PI*x),cos(2.0*M_PI*y))-1.0;
  f(0,0) = -5.0*M_PI*M_PI*sin(M_PI*x)*cos(2.0*M_PI*y);
  ref(0,0) = sin(M_PI*x)*cos(2.0*M_PI*y);


        }
    }

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c1, &__t1);
        block->instance->OPS_kernels[0].time += __t1 - __t2;
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
        block->instance->OPS_kernels[0].mpi_time += __t2 - __t1;
        block->instance->OPS_kernels[0].transfer += ops_compute_transfer(dim, start, end, &arg3);
        block->instance->OPS_kernels[0].transfer += ops_compute_transfer(dim, start, end, &arg4);
        block->instance->OPS_kernels[0].transfer += ops_compute_transfer(dim, start, end, &arg5);
    }
}

#ifdef OPS_LAZY
void ops_par_loop_poisson_kernel_populate(
    const char *name,
    ops_block block,
    int dim,
    int* range,
    ops_arg arg0,
    ops_arg arg1,
    ops_arg arg2,
    ops_arg arg3,
    ops_arg arg4,
    ops_arg arg5
)
{
    ops_arg args[6];

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;
    args[3] = arg3;
    args[4] = arg4;
    args[5] = arg5;

    create_kerneldesc_and_enque("poisson_kernel_populate", args, 6, 0, dim, 1, range, block, ops_par_loop_poisson_kernel_populate_execute);
}
#endif
