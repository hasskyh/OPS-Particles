// Auto-generated at 2026-10-29 00:14:12.308597 by ops-translator legacy


//  ==================
//  Host stub function
//  ==================
#ifndef OPS_LAZY
void ops_par_loop_timeloop_eqH(
    const char *name,
    ops_block block,
    int dim,
    int* range,
    ops_arg arg0
)
{
#else
void ops_par_loop_timeloop_eqH_execute(ops_kernel_descriptor *desc)
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
    if (!ops_checkpointing_before(args, 1, range, 9)) return;
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timing_realloc(block->instance, 9, "timeloop_eqH");
        block->instance->OPS_kernels[9].count++;
        ops_timers_core(&__c1, &__t1);
    }

#ifdef OPS_DEBUG
    ops_register_args(block->instance, args, "timeloop_eqH");
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
    int xdim0_timeloop_eqH = args[0].dat->size[0];
    int ydim0_timeloop_eqH = args[0].dat->size[1];

//  =======================================================
//  Set up initial pointers and exchange halos if necessary
//  =======================================================
    int base0 = args[0].dat->base_offset;
    double * __restrict__ workArray_p = (double *)(args[0].data_d + base0);

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
        block->instance->OPS_kernels[9].mpi_time += __t2 - __t1;
    }

    #pragma omp target teams distribute parallel for collapse(2)
    for (int n_y = start1; n_y < end1; n_y++) {
        for (int n_x = start0; n_x < end0; n_x++) {

#ifdef OPS_SOA
            ACC<double> workArray(9, xdim0_timeloop_eqH, ydim0_timeloop_eqH, workArray_p + n_x*1 + n_y * xdim0_timeloop_eqH*1);
#else
            ACC<double> workArray(9, xdim0_timeloop_eqH, ydim0_timeloop_eqH, workArray_p + 9*(n_x*1 + n_y * xdim0_timeloop_eqH*1));
#endif



    for (int f = 0; f < 9; f++) {
        workArray(f,0,0) = (3.0+4.5*workArray(f,0,0))*workArray(f,0,0);
    }

}
}

if (block->instance->OPS_diags > 1) {
ops_timers_core(&__c1, &__t1);
block->instance->OPS_kernels[9].time += __t1 - __t2;
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
        block->instance->OPS_kernels[9].mpi_time += __t2 - __t1;
        block->instance->OPS_kernels[9].transfer += ops_compute_transfer(dim, start, end, &arg0);
    }
}

#ifdef OPS_LAZY
void ops_par_loop_timeloop_eqH(
    const char *name,
    ops_block block,
    int dim,
    int* range,
    ops_arg arg0
)
{
    ops_arg args[1];

    args[0] = arg0;

    create_kerneldesc_and_enque("timeloop_eqH", args, 1, 9, dim, 1, range, block, ops_par_loop_timeloop_eqH_execute);
}
#endif
