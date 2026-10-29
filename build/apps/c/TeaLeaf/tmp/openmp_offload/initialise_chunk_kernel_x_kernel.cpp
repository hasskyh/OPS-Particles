// Auto-generated at 2026-10-29 00:14:08.616581 by ops-translator legacy


//  ==================
//  Host stub function
//  ==================
#ifndef OPS_LAZY
void ops_par_loop_initialise_chunk_kernel_x(
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
void ops_par_loop_initialise_chunk_kernel_x_execute(ops_kernel_descriptor *desc)
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
    if (!ops_checkpointing_before(args, 3, range, 7)) return;
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timing_realloc(block->instance, 7, "initialise_chunk_kernel_x");
        block->instance->OPS_kernels[7].count++;
        ops_timers_core(&__c1, &__t1);
    }

#ifdef OPS_DEBUG
    ops_register_args(block->instance, args, "initialise_chunk_kernel_x");
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
    int xdim0_initialise_chunk_kernel_x = args[0].dat->size[0];
    int xdim1_initialise_chunk_kernel_x = args[1].dat->size[0];
    int xdim2_initialise_chunk_kernel_x = args[2].dat->size[0];

//  =======================================================
//  Set up initial pointers and exchange halos if necessary
//  =======================================================
    int base0 = args[0].dat->base_offset;
    double * __restrict__ vertexx_p = (double *)(args[0].data_d + base0);

    int base1 = args[1].dat->base_offset;
    int * __restrict__ xx_p = (int *)(args[1].data_d + base1);

    int base2 = args[2].dat->base_offset;
    double * __restrict__ vertexdx_p = (double *)(args[2].data_d + base2);

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
        block->instance->OPS_kernels[7].mpi_time += __t2 - __t1;
    }

    #pragma omp target teams distribute parallel for collapse(2)
    for (int n_y = start1; n_y < end1; n_y++) {
        for (int n_x = start0; n_x < end0; n_x++) {

            ACC<double> vertexx(xdim0_initialise_chunk_kernel_x, vertexx_p + n_x*1 + n_y * xdim0_initialise_chunk_kernel_x*0);

            const ACC<int> xx(xdim1_initialise_chunk_kernel_x, xx_p + n_x*1 + n_y * xdim1_initialise_chunk_kernel_x*0);

            ACC<double> vertexdx(xdim2_initialise_chunk_kernel_x, vertexdx_p + n_x*1 + n_y * xdim2_initialise_chunk_kernel_x*0);

            

  int x_min=field.x_min-2;
  double min_x, d_x;

  d_x = (grid.xmax - grid.xmin)/(double)grid.x_cells;
  min_x=grid.xmin+d_x*field.left;

  vertexx(0,0) = min_x + d_x * (xx(0,0) - x_min);
  vertexdx(0,0) = (double)d_x;

        }
    }

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c1, &__t1);
        block->instance->OPS_kernels[7].time += __t1 - __t2;
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
        block->instance->OPS_kernels[7].mpi_time += __t2 - __t1;
        block->instance->OPS_kernels[7].transfer += ops_compute_transfer(dim, start, end, &arg0);
        block->instance->OPS_kernels[7].transfer += ops_compute_transfer(dim, start, end, &arg1);
        block->instance->OPS_kernels[7].transfer += ops_compute_transfer(dim, start, end, &arg2);
    }
}

#ifdef OPS_LAZY
void ops_par_loop_initialise_chunk_kernel_x(
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

    create_kerneldesc_and_enque("initialise_chunk_kernel_x", args, 3, 7, dim, 1, range, block, ops_par_loop_initialise_chunk_kernel_x_execute);
}
#endif
