// Auto-generated at 2026-10-29 00:14:08.848386 by ops-translator legacy


//  ==================
//  Host stub function
//  ==================
#ifndef OPS_LAZY
void ops_par_loop_tea_leaf_norm2_kernel(
    const char *name,
    ops_block block,
    int dim,
    int* range,
    ops_arg arg0, 
    ops_arg arg1
)
{
#else
void ops_par_loop_tea_leaf_norm2_kernel_execute(ops_kernel_descriptor *desc)
{
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

#if defined(CHECKPOINTING) && !defined(OPS_LAZY)
    if (!ops_checkpointing_before(args, 2, range, 30)) return;
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timing_realloc(block->instance, 30, "tea_leaf_norm2_kernel");
        block->instance->OPS_kernels[30].count++;
        ops_timers_core(&__c1, &__t1);
    }

#ifdef OPS_DEBUG
    ops_register_args(block->instance, args, "tea_leaf_norm2_kernel");
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
    if (compute_ranges(args, 2, block, range, start, end, arg_idx) < 0) return;
#endif

    int start0 = start[0];
    int end0 = end[0];
    int start1 = start[1];
    int end1 = end[1];

    //  =====================================================
    //  Initialize global variable with the dimension of dats
    //  =====================================================
    int xdim0_tea_leaf_norm2_kernel = args[0].dat->size[0];

//  =======================================================
//  Set up initial pointers and exchange halos if necessary
//  =======================================================
    int base0 = args[0].dat->base_offset;
    double * __restrict__ x_p = (double *)(args[0].data_d + base0);

#ifdef OPS_MPI
    double * __restrict__ p_a1 = (double *)(((ops_reduction)args[1].data)->data + ((ops_reduction)args[1].data)->size * block->index);
#else //OPS_MPI
    double * __restrict__ p_a1 = (double *)((ops_reduction)args[1].data)->data;
#endif //OPS_MPI


    int consts_bytes = 0;


    reallocConstArrays(block->instance,consts_bytes);
    consts_bytes = 0;

    mvConstArraysToDevice(block->instance,consts_bytes);


#ifndef OPS_LAZY
//  ==============
//  Halo Exchanges
//  ==============
    ops_H_D_exchanges_device(args, 2);
    ops_halo_exchanges(args, 2,range);
    ops_H_D_exchanges_device(args, 2);
#endif //OPS_LAZY

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[30].mpi_time += __t2 - __t1;
    }

    double p_a1_0 = p_a1[0];
    #pragma omp target teams distribute parallel for collapse(2) reduction(+:p_a1_0)
    for (int n_y = start1; n_y < end1; n_y++) {
        for (int n_x = start0; n_x < end0; n_x++) {

            const ACC<double> x(xdim0_tea_leaf_norm2_kernel, x_p + n_x*1 + n_y * xdim0_tea_leaf_norm2_kernel*1);

            double norm[1];
            norm[0] = ZERO_double;

            
	*norm = *norm + x(0,0)*x(0,0);

            p_a1_0 +=norm[0];
        }
    }

    p_a1[0] = p_a1_0;

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c1, &__t1);
        block->instance->OPS_kernels[30].time += __t1 - __t2;
    }
#ifndef OPS_LAZY
    ops_set_dirtybit_device(args, 2);
#endif

    if (block->instance->OPS_diags > 1) {
        //  ====================
        //  Update kernel record
        //  ====================
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[30].mpi_time += __t2 - __t1;
        block->instance->OPS_kernels[30].transfer += ops_compute_transfer(dim, start, end, &arg0);
    }
}

#ifdef OPS_LAZY
void ops_par_loop_tea_leaf_norm2_kernel(
    const char *name,
    ops_block block,
    int dim,
    int* range,
    ops_arg arg0,
    ops_arg arg1
)
{
    ops_arg args[2];

    args[0] = arg0;
    args[1] = arg1;

    create_kerneldesc_and_enque("tea_leaf_norm2_kernel", args, 2, 30, dim, 1, range, block, ops_par_loop_tea_leaf_norm2_kernel_execute);
}
#endif
