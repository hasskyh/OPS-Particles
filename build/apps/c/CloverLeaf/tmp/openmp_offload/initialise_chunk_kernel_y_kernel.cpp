// Auto-generated at 2026-10-29 00:13:48.647311 by ops-translator legacy


//  ==================
//  Host stub function
//  ==================
#ifndef OPS_LAZY
void ops_par_loop_initialise_chunk_kernel_y(
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
void ops_par_loop_initialise_chunk_kernel_y_execute(ops_kernel_descriptor *desc)
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
    if (!ops_checkpointing_before(args, 3, range, 35)) return;
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timing_realloc(block->instance, 35, "initialise_chunk_kernel_y");
        block->instance->OPS_kernels[35].count++;
        ops_timers_core(&__c1, &__t1);
    }

#ifdef OPS_DEBUG
    ops_register_args(block->instance, args, "initialise_chunk_kernel_y");
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
    int xdim0_initialise_chunk_kernel_y = args[0].dat->size[0];
    int xdim1_initialise_chunk_kernel_y = args[1].dat->size[0];
    int xdim2_initialise_chunk_kernel_y = args[2].dat->size[0];

//  =======================================================
//  Set up initial pointers and exchange halos if necessary
//  =======================================================
    int base0 = args[0].dat->base_offset;
    double * __restrict__ vertexy_p = (double *)(args[0].data_d + base0);

    int base1 = args[1].dat->base_offset;
    int * __restrict__ yy_p = (int *)(args[1].data_d + base1);

    int base2 = args[2].dat->base_offset;
    double * __restrict__ vertexdy_p = (double *)(args[2].data_d + base2);

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
        block->instance->OPS_kernels[35].mpi_time += __t2 - __t1;
    }

    #pragma omp target teams distribute parallel for collapse(2)
    for (int n_y = start1; n_y < end1; n_y++) {
        for (int n_x = start0; n_x < end0; n_x++) {

            ACC<double> vertexy(xdim0_initialise_chunk_kernel_y, vertexy_p + n_x*0 + n_y * xdim0_initialise_chunk_kernel_y*1);

            const ACC<int> yy(xdim1_initialise_chunk_kernel_y, yy_p + n_x*0 + n_y * xdim1_initialise_chunk_kernel_y*1);

            ACC<double> vertexdy(xdim2_initialise_chunk_kernel_y, vertexdy_p + n_x*0 + n_y * xdim2_initialise_chunk_kernel_y*1);

            

  int y_min=field.y_min-2;
  double min_y, d_y;

  d_y = (grid.ymax - grid.ymin)/(double)grid.y_cells;
  min_y=grid.ymin+d_y*field.bottom;

  vertexy(0,0) = min_y + d_y * (yy(0,0) - y_min);
  vertexdy(0,0) = (double)d_y;

        }
    }

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c1, &__t1);
        block->instance->OPS_kernels[35].time += __t1 - __t2;
    }
#ifndef OPS_LAZY
    ops_set_dirtybit_device(args, 3);
    ops_set_halo_dirtybit3(&args[0], range);
    ops_set_halo_dirtybit3(&args[2], range);
#endif

    if (block->instance->OPS_diags > 1) {
        //  ====================
        //  Update kernel record
        //  ====================
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[35].mpi_time += __t2 - __t1;
        block->instance->OPS_kernels[35].transfer += ops_compute_transfer(dim, start, end, &arg0);
        block->instance->OPS_kernels[35].transfer += ops_compute_transfer(dim, start, end, &arg1);
        block->instance->OPS_kernels[35].transfer += ops_compute_transfer(dim, start, end, &arg2);
    }
}

#ifdef OPS_LAZY
void ops_par_loop_initialise_chunk_kernel_y(
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

    create_kerneldesc_and_enque("initialise_chunk_kernel_y", args, 3, 35, dim, 1, range, block, ops_par_loop_initialise_chunk_kernel_y_execute);
}
#endif
