// Auto-generated at 2026-10-29 00:14:11.392469 by ops-translator legacy


//  ==================
//  Host stub function
//  ==================
#ifndef OPS_LAZY
void ops_par_loop_checkop_kernel(
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
void ops_par_loop_checkop_kernel_execute(ops_kernel_descriptor *desc)
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
    if (!ops_checkpointing_before(args, 6, range, 14)) return;
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timing_realloc(block->instance, 14, "checkop_kernel");
        block->instance->OPS_kernels[14].count++;
        ops_timers_core(&__c1, &__t1);
    }

#ifdef OPS_DEBUG
    ops_register_args(block->instance, args, "checkop_kernel");
#endif


//  =================================================
//  compute locally allocated range for the sub-block
//  =================================================
    int start[1];
    int end[1];
#if defined(OPS_MPI) && !defined(OPS_LAZY)
    int arg_idx[1];
#endif

#if defined(OPS_LAZY) || !defined(OPS_MPI)
    for (int n = 0; n < 1; n++) {
        start[n] = range[2*n];
        end[n]   = range[2*n+1];
    }
#else
    if (compute_ranges(args, 6, block, range, start, end, arg_idx) < 0) return;
#endif

    int start0 = start[0];
    int end0 = end[0];

    //  =====================================================
    //  Initialize global variable with the dimension of dats
    //  =====================================================

//  =======================================================
//  Set up initial pointers and exchange halos if necessary
//  =======================================================
    int base0 = args[0].dat->base_offset;
    double * __restrict__ rho_new_p = (double *)(args[0].data_d + base0);

    int base1 = args[1].dat->base_offset;
    double * __restrict__ x_p = (double *)(args[1].data_d + base1);

    int base2 = args[2].dat->base_offset;
    double * __restrict__ rhoin_p = (double *)(args[2].data_d + base2);

#ifdef OPS_MPI
    double * __restrict__ p_a3 = (double *)(((ops_reduction)args[3].data)->data + ((ops_reduction)args[3].data)->size * block->index);
#else //OPS_MPI
    double * __restrict__ p_a3 = (double *)((ops_reduction)args[3].data)->data;
#endif //OPS_MPI

#ifdef OPS_MPI
    double * __restrict__ p_a4 = (double *)(((ops_reduction)args[4].data)->data + ((ops_reduction)args[4].data)->size * block->index);
#else //OPS_MPI
    double * __restrict__ p_a4 = (double *)((ops_reduction)args[4].data)->data;
#endif //OPS_MPI

#ifdef OPS_MPI
    int * __restrict__ p_a5 = (int *)(((ops_reduction)args[5].data)->data + ((ops_reduction)args[5].data)->size * block->index);
#else //OPS_MPI
    int * __restrict__ p_a5 = (int *)((ops_reduction)args[5].data)->data;
#endif //OPS_MPI


    int consts_bytes = 0;


    reallocConstArrays(block->instance,consts_bytes);
    consts_bytes = 0;

    mvConstArraysToDevice(block->instance,consts_bytes);


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
        block->instance->OPS_kernels[14].mpi_time += __t2 - __t1;
    }

    double p_a3_0 = p_a3[0];
    double p_a4_0 = p_a4[0];
    int p_a5_0 = p_a5[0];
    #pragma omp target teams distribute parallel for collapse(1) reduction(+:p_a3_0) reduction(+:p_a4_0) reduction(+:p_a5_0)
    for (int n_x = start0; n_x < end0; n_x++) {

        const ACC<double> rho_new(rho_new_p + n_x*1);

        const ACC<double> x(x_p + n_x*1);

        const ACC<double> rhoin(rhoin_p + n_x*1);

        double pre[1];
        pre[0] = ZERO_double;

        double post[1];
        post[0] = ZERO_double;

        int num[1];
        num[0] = ZERO_int;

        
  double diff;
  diff = (rho_new(0) - rhoin(0));
  if(fabs(diff)<0.01 && x(0) > -4.1){
    *post = *post + diff*diff;
    *num = *num + 1;

  }
  else
    *pre = *pre + (rho_new(0) - rhol)* (rho_new(0) - rhol);

        p_a3_0 +=pre[0];
        p_a4_0 +=post[0];
        p_a5_0 +=num[0];
    }

    p_a3[0] = p_a3_0;

    p_a4[0] = p_a4_0;

    p_a5[0] = p_a5_0;

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c1, &__t1);
        block->instance->OPS_kernels[14].time += __t1 - __t2;
    }
#ifndef OPS_LAZY
    ops_set_dirtybit_device(args, 6);
#endif

    if (block->instance->OPS_diags > 1) {
        //  ====================
        //  Update kernel record
        //  ====================
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[14].mpi_time += __t2 - __t1;
        block->instance->OPS_kernels[14].transfer += ops_compute_transfer(dim, start, end, &arg0);
        block->instance->OPS_kernels[14].transfer += ops_compute_transfer(dim, start, end, &arg1);
        block->instance->OPS_kernels[14].transfer += ops_compute_transfer(dim, start, end, &arg2);
    }
}

#ifdef OPS_LAZY
void ops_par_loop_checkop_kernel(
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

    create_kerneldesc_and_enque("checkop_kernel", args, 6, 14, dim, 1, range, block, ops_par_loop_checkop_kernel_execute);
}
#endif
