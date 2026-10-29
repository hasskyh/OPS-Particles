// Auto-generated at 2026-10-29 00:14:08.604025 by ops-translator legacy


//  ==================
//  Host stub function
//  ==================
#ifndef OPS_LAZY
void ops_par_loop_initialise_chunk_kernel_zero(
    const char *name,
    ops_block block,
    int dim,
    int* range,
    ops_arg arg0
)
{
#else
void ops_par_loop_initialise_chunk_kernel_zero_execute(ops_kernel_descriptor *desc)
{
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

#if defined(CHECKPOINTING) && !defined(OPS_LAZY)
    if (!ops_checkpointing_before(args, 1, range, 2)) return;
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timing_realloc(block->instance, 2, "initialise_chunk_kernel_zero");
        block->instance->OPS_kernels[2].count++;
        ops_timers_core(&__c1, &__t1);
    }

#ifdef OPS_DEBUG
    ops_register_args(block->instance, args, "initialise_chunk_kernel_zero");
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
    if (compute_ranges(args, 1, block, range, start, end, arg_idx) < 0) return;
#endif

    int start0 = start[0];
    int end0 = end[0];
    int start1 = start[1];
    int end1 = end[1];

    //  =====================================================
    //  Initialize global variable with the dimension of dats
    //  =====================================================
    int xdim0_initialise_chunk_kernel_zero = args[0].dat->size[0];

//  =======================================================
//  Set up initial pointers and exchange halos if necessary
//  =======================================================
    int base0 = args[0].dat->base_offset;
    double * __restrict__ var_p = (double *)(args[0].data_d + base0);

#ifndef OPS_LAZY
//  ==============
//  Halo Exchanges
//  ==============
    ops_H_D_exchanges_device(args, 1);
    ops_halo_exchanges(args, 1,range);
    ops_H_D_exchanges_device(args, 1);
#endif //OPS_LAZY

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[2].mpi_time += __t2 - __t1;
    }

    #pragma omp target teams distribute parallel for collapse(2)
    for (int n_y = start1; n_y < end1; n_y++) {
        for (int n_x = start0; n_x < end0; n_x++) {

            ACC<double> var(xdim0_initialise_chunk_kernel_zero, var_p + n_x*1 + n_y * xdim0_initialise_chunk_kernel_zero*1);

            
  var(0,0) = 0.0;

        }
    }

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c1, &__t1);
        block->instance->OPS_kernels[2].time += __t1 - __t2;
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
        block->instance->OPS_kernels[2].mpi_time += __t2 - __t1;
        block->instance->OPS_kernels[2].transfer += ops_compute_transfer(dim, start, end, &arg0);
    }
}

#ifdef OPS_LAZY
void ops_par_loop_initialise_chunk_kernel_zero(
    const char *name,
    ops_block block,
    int dim,
    int* range,
    ops_arg arg0
)
{
    ops_arg args[1];

    args[0] = arg0;

    create_kerneldesc_and_enque("initialise_chunk_kernel_zero", args, 1, 2, dim, 1, range, block, ops_par_loop_initialise_chunk_kernel_zero_execute);
}
#endif
