// Auto-generated at 2026-10-29 00:13:57.468828 by ops-translator legacy


//  ==================
//  Host stub function
//  ==================
#ifndef OPS_LAZY
void ops_par_loop_flux_calc_kernely(
    const char *name,
    ops_block block,
    int dim,
    int* range,
    ops_arg arg0, 
    ops_arg arg1, 
    ops_arg arg2, 
    ops_arg arg3
)
{
#else
void ops_par_loop_flux_calc_kernely_execute(ops_kernel_descriptor *desc)
{
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

#if defined(CHECKPOINTING) && !defined(OPS_LAZY)
    if (!ops_checkpointing_before(args, 4, range, 39)) return;
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timing_realloc(block->instance, 39, "flux_calc_kernely");
        block->instance->OPS_kernels[39].count++;
        ops_timers_core(&__c1, &__t1);
    }

#ifdef OPS_DEBUG
    ops_register_args(block->instance, args, "flux_calc_kernely");
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
    if (compute_ranges(args, 4, block, range, start, end, arg_idx) < 0) return;
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
    int xdim0_flux_calc_kernely = args[0].dat->size[0];
    int ydim0_flux_calc_kernely = args[0].dat->size[1];
    int xdim1_flux_calc_kernely = args[1].dat->size[0];
    int ydim1_flux_calc_kernely = args[1].dat->size[1];
    int xdim2_flux_calc_kernely = args[2].dat->size[0];
    int ydim2_flux_calc_kernely = args[2].dat->size[1];
    int xdim3_flux_calc_kernely = args[3].dat->size[0];
    int ydim3_flux_calc_kernely = args[3].dat->size[1];

//  =======================================================
//  Set up initial pointers and exchange halos if necessary
//  =======================================================
    int base0 = args[0].dat->base_offset;
    double * __restrict__ vol_flux_y_p = (double *)(args[0].data_d + base0);

    int base1 = args[1].dat->base_offset;
    double * __restrict__ yarea_p = (double *)(args[1].data_d + base1);

    int base2 = args[2].dat->base_offset;
    double * __restrict__ yvel0_p = (double *)(args[2].data_d + base2);

    int base3 = args[3].dat->base_offset;
    double * __restrict__ yvel1_p = (double *)(args[3].data_d + base3);

#ifndef OPS_LAZY
//  ==============
//  Halo Exchanges
//  ==============
    ops_H_D_exchanges_device(args, 4);
    ops_halo_exchanges(args, 4,range);
    ops_H_D_exchanges_device(args, 4);
#endif //OPS_LAZY

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[39].mpi_time += __t2 - __t1;
    }

    #pragma omp target teams distribute parallel for collapse(3)
    for (int n_z = start2; n_z < end2; n_z++) {
        for (int n_y = start1; n_y < end1; n_y++) {
            for (int n_x = start0; n_x < end0; n_x++) {

                ACC<double> vol_flux_y(xdim0_flux_calc_kernely, ydim0_flux_calc_kernely, vol_flux_y_p + n_x*1 + n_y * xdim0_flux_calc_kernely*1 + n_z * xdim0_flux_calc_kernely * ydim0_flux_calc_kernely*1);

                const ACC<double> yarea(xdim1_flux_calc_kernely, ydim1_flux_calc_kernely, yarea_p + n_x*1 + n_y * xdim1_flux_calc_kernely*1 + n_z * xdim1_flux_calc_kernely * ydim1_flux_calc_kernely*1);

                const ACC<double> yvel0(xdim2_flux_calc_kernely, ydim2_flux_calc_kernely, yvel0_p + n_x*1 + n_y * xdim2_flux_calc_kernely*1 + n_z * xdim2_flux_calc_kernely * ydim2_flux_calc_kernely*1);

                const ACC<double> yvel1(xdim3_flux_calc_kernely, ydim3_flux_calc_kernely, yvel1_p + n_x*1 + n_y * xdim3_flux_calc_kernely*1 + n_z * xdim3_flux_calc_kernely * ydim3_flux_calc_kernely*1);

                

  vol_flux_y(0,0,0) = 0.125 * dt * (yarea(0,0,0)) *
  ( yvel0(0,0,0) + yvel0(1,0,0) + yvel0(0,0,1) + yvel0(1,0,1) +
    yvel1(0,0,0) + yvel1(1,0,0) + yvel1(0,0,1) + yvel1(1,0,1));

            }
        }
    }

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c1, &__t1);
        block->instance->OPS_kernels[39].time += __t1 - __t2;
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
        block->instance->OPS_kernels[39].mpi_time += __t2 - __t1;
        block->instance->OPS_kernels[39].transfer += ops_compute_transfer(dim, start, end, &arg0);
        block->instance->OPS_kernels[39].transfer += ops_compute_transfer(dim, start, end, &arg1);
        block->instance->OPS_kernels[39].transfer += ops_compute_transfer(dim, start, end, &arg2);
        block->instance->OPS_kernels[39].transfer += ops_compute_transfer(dim, start, end, &arg3);
    }
}

#ifdef OPS_LAZY
void ops_par_loop_flux_calc_kernely(
    const char *name,
    ops_block block,
    int dim,
    int* range,
    ops_arg arg0,
    ops_arg arg1,
    ops_arg arg2,
    ops_arg arg3
)
{
    ops_arg args[4];

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;
    args[3] = arg3;

    create_kerneldesc_and_enque("flux_calc_kernely", args, 4, 39, dim, 1, range, block, ops_par_loop_flux_calc_kernely_execute);
}
#endif
