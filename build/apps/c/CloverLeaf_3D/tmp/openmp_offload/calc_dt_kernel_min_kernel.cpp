// Auto-generated at 2026-10-29 00:13:57.410324 by ops-translator legacy


//  ==================
//  Host stub function
//  ==================
#ifndef OPS_LAZY
void ops_par_loop_calc_dt_kernel_min(
    const char *name,
    ops_block block,
    int dim,
    int* range,
    ops_arg arg0, 
    ops_arg arg1
)
{
#else
void ops_par_loop_calc_dt_kernel_min_execute(ops_kernel_descriptor *desc)
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
    if (!ops_checkpointing_before(args, 2, range, 34)) return;
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timing_realloc(block->instance, 34, "calc_dt_kernel_min");
        block->instance->OPS_kernels[34].count++;
        ops_timers_core(&__c1, &__t1);
    }

#ifdef OPS_DEBUG
    ops_register_args(block->instance, args, "calc_dt_kernel_min");
#endif


//  =================================================
//  compute locally allocated range for the sub-block
//  =================================================
    int start[3];
    int end[3];
#if defined(OPS_MPI) && !defined(OPS_LAZY)
    int arg_idx[3];
#endif

#if defined(OPS_LAZY) || !defined(OPS_MPI)
    for (int n = 0; n < 3; n++) {
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
    int start2 = start[2];
    int end2 = end[2];

    //  =====================================================
    //  Initialize global variable with the dimension of dats
    //  =====================================================
    int xdim0_calc_dt_kernel_min = args[0].dat->size[0];
    int ydim0_calc_dt_kernel_min = args[0].dat->size[1];

//  =======================================================
//  Set up initial pointers and exchange halos if necessary
//  =======================================================
    int base0 = args[0].dat->base_offset;
    double * __restrict__ dt_min_p = (double *)(args[0].data_d + base0);

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
        block->instance->OPS_kernels[34].mpi_time += __t2 - __t1;
    }

    double p_a1_0 = p_a1[0];
    #pragma omp target teams distribute parallel for collapse(3) reduction(min:p_a1_0)
    for (int n_z = start2; n_z < end2; n_z++) {
        for (int n_y = start1; n_y < end1; n_y++) {
            for (int n_x = start0; n_x < end0; n_x++) {

                const ACC<double> dt_min(xdim0_calc_dt_kernel_min, ydim0_calc_dt_kernel_min, dt_min_p + n_x*1 + n_y * xdim0_calc_dt_kernel_min*1 + n_z * xdim0_calc_dt_kernel_min * ydim0_calc_dt_kernel_min*1);

                double dt_min_val[1];
                dt_min_val[0] = INFINITY_double;

                
  *dt_min_val = MIN(*dt_min_val, dt_min(0,0,0));


                p_a1_0 = MIN(p_a1_0,dt_min_val[0]);
            }
        }
    }

    p_a1[0] = p_a1_0;

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c1, &__t1);
        block->instance->OPS_kernels[34].time += __t1 - __t2;
    }
#ifndef OPS_LAZY
    ops_set_dirtybit_device(args, 2);
#endif

    if (block->instance->OPS_diags > 1) {
        //  ====================
        //  Update kernel record
        //  ====================
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[34].mpi_time += __t2 - __t1;
        block->instance->OPS_kernels[34].transfer += ops_compute_transfer(dim, start, end, &arg0);
    }
}

#ifdef OPS_LAZY
void ops_par_loop_calc_dt_kernel_min(
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

    create_kerneldesc_and_enque("calc_dt_kernel_min", args, 2, 34, dim, 1, range, block, ops_par_loop_calc_dt_kernel_min_execute);
}
#endif
