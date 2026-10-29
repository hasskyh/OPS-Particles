// Auto-generated at 2026-10-29 00:13:48.415548 by ops-translator legacy


//  ==================
//  Host stub function
//  ==================
#ifndef OPS_LAZY
void ops_par_loop_advec_cell_kernel1_xdir(
    const char *name,
    ops_block block,
    int dim,
    int* range,
    ops_arg arg0, 
    ops_arg arg1, 
    ops_arg arg2, 
    ops_arg arg3, 
    ops_arg arg4
)
{
#else
void ops_par_loop_advec_cell_kernel1_xdir_execute(ops_kernel_descriptor *desc)
{
    ops_block block = desc->block;
    int dim = desc->dim;
    int *range = desc->range;
    ops_arg arg0 = desc->args[0];
    ops_arg arg1 = desc->args[1];
    ops_arg arg2 = desc->args[2];
    ops_arg arg3 = desc->args[3];
    ops_arg arg4 = desc->args[4];
#endif

//  ======
//  Timing
//  ======
    double __t1, __t2, __c1, __c2;

    ops_arg args[5];

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;
    args[3] = arg3;
    args[4] = arg4;

#if defined(CHECKPOINTING) && !defined(OPS_LAZY)
    if (!ops_checkpointing_before(args, 5, range, 3)) return;
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timing_realloc(block->instance, 3, "advec_cell_kernel1_xdir");
        block->instance->OPS_kernels[3].count++;
        ops_timers_core(&__c1, &__t1);
    }

#ifdef OPS_DEBUG
    ops_register_args(block->instance, args, "advec_cell_kernel1_xdir");
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
    if (compute_ranges(args, 5, block, range, start, end, arg_idx) < 0) return;
#endif

    int start0 = start[0];
    int end0 = end[0];
    int start1 = start[1];
    int end1 = end[1];

    //  =====================================================
    //  Initialize global variable with the dimension of dats
    //  =====================================================
    int xdim0_advec_cell_kernel1_xdir = args[0].dat->size[0];
    int xdim1_advec_cell_kernel1_xdir = args[1].dat->size[0];
    int xdim2_advec_cell_kernel1_xdir = args[2].dat->size[0];
    int xdim3_advec_cell_kernel1_xdir = args[3].dat->size[0];
    int xdim4_advec_cell_kernel1_xdir = args[4].dat->size[0];

//  =======================================================
//  Set up initial pointers and exchange halos if necessary
//  =======================================================
    int base0 = args[0].dat->base_offset;
    double * __restrict__ pre_vol_p = (double *)(args[0].data_d + base0);

    int base1 = args[1].dat->base_offset;
    double * __restrict__ post_vol_p = (double *)(args[1].data_d + base1);

    int base2 = args[2].dat->base_offset;
    double * __restrict__ volume_p = (double *)(args[2].data_d + base2);

    int base3 = args[3].dat->base_offset;
    double * __restrict__ vol_flux_x_p = (double *)(args[3].data_d + base3);

    int base4 = args[4].dat->base_offset;
    double * __restrict__ vol_flux_y_p = (double *)(args[4].data_d + base4);

#ifndef OPS_LAZY
//  ==============
//  Halo Exchanges
//  ==============
    ops_H_D_exchanges_device(args, 5);
    ops_halo_exchanges(args, 5,range);
    ops_H_D_exchanges_device(args, 5);
#endif //OPS_LAZY

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[3].mpi_time += __t2 - __t1;
    }

    #pragma omp target teams distribute parallel for collapse(2)
    for (int n_y = start1; n_y < end1; n_y++) {
        for (int n_x = start0; n_x < end0; n_x++) {

            ACC<double> pre_vol(xdim0_advec_cell_kernel1_xdir, pre_vol_p + n_x*1 + n_y * xdim0_advec_cell_kernel1_xdir*1);

            ACC<double> post_vol(xdim1_advec_cell_kernel1_xdir, post_vol_p + n_x*1 + n_y * xdim1_advec_cell_kernel1_xdir*1);

            const ACC<double> volume(xdim2_advec_cell_kernel1_xdir, volume_p + n_x*1 + n_y * xdim2_advec_cell_kernel1_xdir*1);

            const ACC<double> vol_flux_x(xdim3_advec_cell_kernel1_xdir, vol_flux_x_p + n_x*1 + n_y * xdim3_advec_cell_kernel1_xdir*1);

            const ACC<double> vol_flux_y(xdim4_advec_cell_kernel1_xdir, vol_flux_y_p + n_x*1 + n_y * xdim4_advec_cell_kernel1_xdir*1);

            

  pre_vol(0,0) = volume(0,0) + ( vol_flux_x(1,0) - vol_flux_x(0,0) +
                           vol_flux_y(0,1) - vol_flux_y(0,0));
  post_vol(0,0) = pre_vol(0,0) - ( vol_flux_x(1,0) - vol_flux_x(0,0));


        }
    }

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c1, &__t1);
        block->instance->OPS_kernels[3].time += __t1 - __t2;
    }
#ifndef OPS_LAZY
    ops_set_dirtybit_device(args, 5);
    ops_set_halo_dirtybit3(&args[0], range);
    ops_set_halo_dirtybit3(&args[1], range);
#endif

    if (block->instance->OPS_diags > 1) {
        //  ====================
        //  Update kernel record
        //  ====================
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[3].mpi_time += __t2 - __t1;
        block->instance->OPS_kernels[3].transfer += ops_compute_transfer(dim, start, end, &arg0);
        block->instance->OPS_kernels[3].transfer += ops_compute_transfer(dim, start, end, &arg1);
        block->instance->OPS_kernels[3].transfer += ops_compute_transfer(dim, start, end, &arg2);
        block->instance->OPS_kernels[3].transfer += ops_compute_transfer(dim, start, end, &arg3);
        block->instance->OPS_kernels[3].transfer += ops_compute_transfer(dim, start, end, &arg4);
    }
}

#ifdef OPS_LAZY
void ops_par_loop_advec_cell_kernel1_xdir(
    const char *name,
    ops_block block,
    int dim,
    int* range,
    ops_arg arg0,
    ops_arg arg1,
    ops_arg arg2,
    ops_arg arg3,
    ops_arg arg4
)
{
    ops_arg args[5];

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;
    args[3] = arg3;
    args[4] = arg4;

    create_kerneldesc_and_enque("advec_cell_kernel1_xdir", args, 5, 3, dim, 1, range, block, ops_par_loop_advec_cell_kernel1_xdir_execute);
}
#endif
