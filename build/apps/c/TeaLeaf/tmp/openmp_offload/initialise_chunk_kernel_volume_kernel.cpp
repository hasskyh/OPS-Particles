// Auto-generated at 2026-10-29 00:14:08.638555 by ops-translator legacy


//  ==================
//  Host stub function
//  ==================
#ifndef OPS_LAZY
void ops_par_loop_initialise_chunk_kernel_volume(
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
void ops_par_loop_initialise_chunk_kernel_volume_execute(ops_kernel_descriptor *desc)
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
    if (!ops_checkpointing_before(args, 5, range, 11)) return;
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timing_realloc(block->instance, 11, "initialise_chunk_kernel_volume");
        block->instance->OPS_kernels[11].count++;
        ops_timers_core(&__c1, &__t1);
    }

#ifdef OPS_DEBUG
    ops_register_args(block->instance, args, "initialise_chunk_kernel_volume");
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
    int xdim0_initialise_chunk_kernel_volume = args[0].dat->size[0];
    int xdim1_initialise_chunk_kernel_volume = args[1].dat->size[0];
    int xdim2_initialise_chunk_kernel_volume = args[2].dat->size[0];
    int xdim3_initialise_chunk_kernel_volume = args[3].dat->size[0];
    int xdim4_initialise_chunk_kernel_volume = args[4].dat->size[0];

//  =======================================================
//  Set up initial pointers and exchange halos if necessary
//  =======================================================
    int base0 = args[0].dat->base_offset;
    double * __restrict__ volume_p = (double *)(args[0].data_d + base0);

    int base1 = args[1].dat->base_offset;
    double * __restrict__ celldy_p = (double *)(args[1].data_d + base1);

    int base2 = args[2].dat->base_offset;
    double * __restrict__ xarea_p = (double *)(args[2].data_d + base2);

    int base3 = args[3].dat->base_offset;
    double * __restrict__ celldx_p = (double *)(args[3].data_d + base3);

    int base4 = args[4].dat->base_offset;
    double * __restrict__ yarea_p = (double *)(args[4].data_d + base4);

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
        block->instance->OPS_kernels[11].mpi_time += __t2 - __t1;
    }

    #pragma omp target teams distribute parallel for collapse(2)
    for (int n_y = start1; n_y < end1; n_y++) {
        for (int n_x = start0; n_x < end0; n_x++) {

            ACC<double> volume(xdim0_initialise_chunk_kernel_volume, volume_p + n_x*1 + n_y * xdim0_initialise_chunk_kernel_volume*1);

            const ACC<double> celldy(xdim1_initialise_chunk_kernel_volume, celldy_p + n_x*0 + n_y * xdim1_initialise_chunk_kernel_volume*1);

            ACC<double> xarea(xdim2_initialise_chunk_kernel_volume, xarea_p + n_x*1 + n_y * xdim2_initialise_chunk_kernel_volume*1);

            const ACC<double> celldx(xdim3_initialise_chunk_kernel_volume, celldx_p + n_x*1 + n_y * xdim3_initialise_chunk_kernel_volume*0);

            ACC<double> yarea(xdim4_initialise_chunk_kernel_volume, yarea_p + n_x*1 + n_y * xdim4_initialise_chunk_kernel_volume*1);

            

  double d_x, d_y;

  d_x = (grid.xmax - grid.xmin)/(double)grid.x_cells;
  d_y = (grid.ymax - grid.ymin)/(double)grid.y_cells;

  volume(0,0) = d_x*d_y;
  xarea(0,0) = celldy(0,0);
  yarea(0,0) = celldx(0,0);

        }
    }

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c1, &__t1);
        block->instance->OPS_kernels[11].time += __t1 - __t2;
    }
#ifndef OPS_LAZY
    ops_set_dirtybit_device(args, 5);
    ops_set_halo_dirtybit3(&args[0], range);
    ops_set_halo_dirtybit3(&args[2], range);
    ops_set_halo_dirtybit3(&args[4], range);
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
        block->instance->OPS_kernels[11].transfer += ops_compute_transfer(dim, start, end, &arg4);
    }
}

#ifdef OPS_LAZY
void ops_par_loop_initialise_chunk_kernel_volume(
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

    create_kerneldesc_and_enque("initialise_chunk_kernel_volume", args, 5, 11, dim, 1, range, block, ops_par_loop_initialise_chunk_kernel_volume_execute);
}
#endif
