// Auto-generated at 2026-10-29 00:14:08.662261 by ops-translator legacy


//  ==================
//  Host stub function
//  ==================
#ifndef OPS_LAZY
void ops_par_loop_tea_leaf_dot_kernel(
    const char *name,
    ops_block block,
    int dim,
    int* range,
    ops_arg arg0, 
    ops_arg arg1, 
    ops_arg arg2
)
{
#else
void ops_par_loop_tea_leaf_dot_kernel_execute(ops_kernel_descriptor *desc)
{
    ops_block block = desc->block;
    int dim = desc->dim;
    int *range = desc->range;
    ops_arg arg0 = desc->args[0];
    ops_arg arg1 = desc->args[1];
    ops_arg arg2 = desc->args[2];
#endif

//  ======
//  Timing
//  ======
    double __t1, __t2, __c1, __c2;

    ops_arg args[3];

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;

#if defined(CHECKPOINTING) && !defined(OPS_LAZY)
    if (!ops_checkpointing_before(args, 3, range, 15)) return;
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timing_realloc(block->instance, 15, "tea_leaf_dot_kernel");
        block->instance->OPS_kernels[15].count++;
        ops_timers_core(&__c1, &__t1);
    }

#ifdef OPS_DEBUG
    ops_register_args(block->instance, args, "tea_leaf_dot_kernel");
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
    if (compute_ranges(args, 3, block, range, start, end, arg_idx) < 0) return;
#endif

    int start0 = start[0];
    int end0 = end[0];
    int start1 = start[1];
    int end1 = end[1];

    //  =====================================================
    //  Initialize global variable with the dimension of dats
    //  =====================================================
    int xdim0_tea_leaf_dot_kernel = args[0].dat->size[0];
    int xdim1_tea_leaf_dot_kernel = args[1].dat->size[0];

//  =======================================================
//  Set up initial pointers and exchange halos if necessary
//  =======================================================
    int base0 = args[0].dat->base_offset;
    double * __restrict__ r_p = (double *)(args[0].data_d + base0);

    int base1 = args[1].dat->base_offset;
    double * __restrict__ p_p = (double *)(args[1].data_d + base1);

#ifdef OPS_MPI
    double * __restrict__ p_a2 = (double *)(((ops_reduction)args[2].data)->data + ((ops_reduction)args[2].data)->size * block->index);
#else //OPS_MPI
    double * __restrict__ p_a2 = (double *)((ops_reduction)args[2].data)->data;
#endif //OPS_MPI


    int consts_bytes = 0;


    reallocConstArrays(block->instance,consts_bytes);
    consts_bytes = 0;

    mvConstArraysToDevice(block->instance,consts_bytes);


#ifndef OPS_LAZY
//  ==============
//  Halo Exchanges
//  ==============
    ops_H_D_exchanges_device(args, 3);
    ops_halo_exchanges(args, 3,range);
    ops_H_D_exchanges_device(args, 3);
#endif //OPS_LAZY

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[15].mpi_time += __t2 - __t1;
    }

    double p_a2_0 = p_a2[0];
    #pragma omp target teams distribute parallel for collapse(2) reduction(+:p_a2_0)
    for (int n_y = start1; n_y < end1; n_y++) {
        for (int n_x = start0; n_x < end0; n_x++) {

            const ACC<double> r(xdim0_tea_leaf_dot_kernel, r_p + n_x*1 + n_y * xdim0_tea_leaf_dot_kernel*1);

            const ACC<double> p(xdim1_tea_leaf_dot_kernel, p_p + n_x*1 + n_y * xdim1_tea_leaf_dot_kernel*1);

            double rro[1];
            rro[0] = ZERO_double;

            
  *rro = *rro + r(0,0) * p(0,0);

            p_a2_0 +=rro[0];
        }
    }

    p_a2[0] = p_a2_0;

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c1, &__t1);
        block->instance->OPS_kernels[15].time += __t1 - __t2;
    }
#ifndef OPS_LAZY
    ops_set_dirtybit_device(args, 3);
#endif

    if (block->instance->OPS_diags > 1) {
        //  ====================
        //  Update kernel record
        //  ====================
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[15].mpi_time += __t2 - __t1;
        block->instance->OPS_kernels[15].transfer += ops_compute_transfer(dim, start, end, &arg0);
        block->instance->OPS_kernels[15].transfer += ops_compute_transfer(dim, start, end, &arg1);
    }
}

#ifdef OPS_LAZY
void ops_par_loop_tea_leaf_dot_kernel(
    const char *name,
    ops_block block,
    int dim,
    int* range,
    ops_arg arg0,
    ops_arg arg1,
    ops_arg arg2
)
{
    ops_arg args[3];

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;

    create_kerneldesc_and_enque("tea_leaf_dot_kernel", args, 3, 15, dim, 1, range, block, ops_par_loop_tea_leaf_dot_kernel_execute);
}
#endif
