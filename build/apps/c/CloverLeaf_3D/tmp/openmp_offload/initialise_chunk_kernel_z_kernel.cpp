// Auto-generated at 2026-10-29 00:13:57.582497 by ops-translator legacy


//  ==================
//  Host stub function
//  ==================
#ifndef OPS_LAZY
void ops_par_loop_initialise_chunk_kernel_z(
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
void ops_par_loop_initialise_chunk_kernel_z_execute(ops_kernel_descriptor *desc)
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
    if (!ops_checkpointing_before(args, 3, range, 48)) return;
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timing_realloc(block->instance, 48, "initialise_chunk_kernel_z");
        block->instance->OPS_kernels[48].count++;
        ops_timers_core(&__c1, &__t1);
    }

#ifdef OPS_DEBUG
    ops_register_args(block->instance, args, "initialise_chunk_kernel_z");
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
    if (compute_ranges(args, 3, block, range, start, end, arg_idx) < 0) return;
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
    int xdim0_initialise_chunk_kernel_z = args[0].dat->size[0];
    int ydim0_initialise_chunk_kernel_z = args[0].dat->size[1];
    int xdim1_initialise_chunk_kernel_z = args[1].dat->size[0];
    int ydim1_initialise_chunk_kernel_z = args[1].dat->size[1];
    int xdim2_initialise_chunk_kernel_z = args[2].dat->size[0];
    int ydim2_initialise_chunk_kernel_z = args[2].dat->size[1];

//  =======================================================
//  Set up initial pointers and exchange halos if necessary
//  =======================================================
    int base0 = args[0].dat->base_offset;
    double * __restrict__ vertexz_p = (double *)(args[0].data_d + base0);

    int base1 = args[1].dat->base_offset;
    int * __restrict__ zz_p = (int *)(args[1].data_d + base1);

    int base2 = args[2].dat->base_offset;
    double * __restrict__ vertexdz_p = (double *)(args[2].data_d + base2);

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
        block->instance->OPS_kernels[48].mpi_time += __t2 - __t1;
    }

    #pragma omp target teams distribute parallel for collapse(3)
    for (int n_z = start2; n_z < end2; n_z++) {
        for (int n_y = start1; n_y < end1; n_y++) {
            for (int n_x = start0; n_x < end0; n_x++) {

                ACC<double> vertexz(xdim0_initialise_chunk_kernel_z, ydim0_initialise_chunk_kernel_z, vertexz_p + n_x*0 + n_y * xdim0_initialise_chunk_kernel_z*0 + n_z * xdim0_initialise_chunk_kernel_z * ydim0_initialise_chunk_kernel_z*1);

                const ACC<int> zz(xdim1_initialise_chunk_kernel_z, ydim1_initialise_chunk_kernel_z, zz_p + n_x*0 + n_y * xdim1_initialise_chunk_kernel_z*0 + n_z * xdim1_initialise_chunk_kernel_z * ydim1_initialise_chunk_kernel_z*1);

                ACC<double> vertexdz(xdim2_initialise_chunk_kernel_z, ydim2_initialise_chunk_kernel_z, vertexdz_p + n_x*0 + n_y * xdim2_initialise_chunk_kernel_z*0 + n_z * xdim2_initialise_chunk_kernel_z * ydim2_initialise_chunk_kernel_z*1);

                
  int z_min=field.z_min-2;

  double min_z, d_z;
  d_z = (grid.zmax - grid.zmin)/(double)grid.z_cells;
  min_z=grid.zmin+d_z*field.back;

  vertexz(0,0,0) = min_z + d_z * (zz(0,0,0) - z_min);
  vertexdz(0,0,0) = (double)d_z;

            }
        }
    }

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c1, &__t1);
        block->instance->OPS_kernels[48].time += __t1 - __t2;
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
        block->instance->OPS_kernels[48].mpi_time += __t2 - __t1;
        block->instance->OPS_kernels[48].transfer += ops_compute_transfer(dim, start, end, &arg0);
        block->instance->OPS_kernels[48].transfer += ops_compute_transfer(dim, start, end, &arg1);
        block->instance->OPS_kernels[48].transfer += ops_compute_transfer(dim, start, end, &arg2);
    }
}

#ifdef OPS_LAZY
void ops_par_loop_initialise_chunk_kernel_z(
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

    create_kerneldesc_and_enque("initialise_chunk_kernel_z", args, 3, 48, dim, 1, range, block, ops_par_loop_initialise_chunk_kernel_z_execute);
}
#endif
