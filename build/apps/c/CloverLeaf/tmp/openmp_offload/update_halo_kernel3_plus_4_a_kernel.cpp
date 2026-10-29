// Auto-generated at 2026-10-29 00:13:48.920062 by ops-translator legacy


//  ==================
//  Host stub function
//  ==================
#ifndef OPS_LAZY
void ops_par_loop_update_halo_kernel3_plus_4_a(
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
void ops_par_loop_update_halo_kernel3_plus_4_a_execute(ops_kernel_descriptor *desc)
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
    if (!ops_checkpointing_before(args, 3, range, 66)) return;
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timing_realloc(block->instance, 66, "update_halo_kernel3_plus_4_a");
        block->instance->OPS_kernels[66].count++;
        ops_timers_core(&__c1, &__t1);
    }

#ifdef OPS_DEBUG
    ops_register_args(block->instance, args, "update_halo_kernel3_plus_4_a");
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
    int xdim0_update_halo_kernel3_plus_4_a = args[0].dat->size[0];
    int xdim1_update_halo_kernel3_plus_4_a = args[1].dat->size[0];

//  =======================================================
//  Set up initial pointers and exchange halos if necessary
//  =======================================================
    int base0 = args[0].dat->base_offset;
    double * __restrict__ vol_flux_x_p = (double *)(args[0].data_d + base0);

    int base1 = args[1].dat->base_offset;
    double * __restrict__ mass_flux_x_p = (double *)(args[1].data_d + base1);



    int consts_bytes = 0;

    int *arg2h = (int *)args[2].data;
    consts_bytes += ROUND_UP(args[2].dim*sizeof(int));

    reallocConstArrays(block->instance,consts_bytes);
    consts_bytes = 0;

    args[2].data = block->instance->OPS_consts_h + consts_bytes;
    args[2].data_d = block->instance->OPS_consts_d + consts_bytes;
    for (int d = 0; d < args[2].dim; d++) {
        ((int *)args[2].data)[d] = arg2h[d];
    }
    consts_bytes += ROUND_UP(args[2].dim*sizeof(int));

    mvConstArraysToDevice(block->instance,consts_bytes);

    int * __restrict__ fields = (int *)args[2].data_d;

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
        block->instance->OPS_kernels[66].mpi_time += __t2 - __t1;
    }

    #pragma omp target teams distribute parallel for collapse(2)
    for (int n_y = start1; n_y < end1; n_y++) {
        for (int n_x = start0; n_x < end0; n_x++) {

            ACC<double> vol_flux_x(xdim0_update_halo_kernel3_plus_4_a, vol_flux_x_p + n_x*1 + n_y * xdim0_update_halo_kernel3_plus_4_a*1);

            ACC<double> mass_flux_x(xdim1_update_halo_kernel3_plus_4_a, mass_flux_x_p + n_x*1 + n_y * xdim1_update_halo_kernel3_plus_4_a*1);

            
  if(fields[FIELD_VOL_FLUX_X] == 1)  vol_flux_x(0,0)  = vol_flux_x(0,4);
  if(fields[FIELD_MASS_FLUX_X] == 1) mass_flux_x(0,0) = mass_flux_x(0,4);

        }
    }

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c1, &__t1);
        block->instance->OPS_kernels[66].time += __t1 - __t2;
    }
#ifndef OPS_LAZY
    ops_set_dirtybit_device(args, 3);
    ops_set_halo_dirtybit3(&args[0], range);
    ops_set_halo_dirtybit3(&args[1], range);
#endif

    if (block->instance->OPS_diags > 1) {
        //  ====================
        //  Update kernel record
        //  ====================
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[66].mpi_time += __t2 - __t1;
        block->instance->OPS_kernels[66].transfer += ops_compute_transfer(dim, start, end, &arg0);
        block->instance->OPS_kernels[66].transfer += ops_compute_transfer(dim, start, end, &arg1);
    }
}

#ifdef OPS_LAZY
void ops_par_loop_update_halo_kernel3_plus_4_a(
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

    create_kerneldesc_and_enque("update_halo_kernel3_plus_4_a", args, 3, 66, dim, 1, range, block, ops_par_loop_update_halo_kernel3_plus_4_a_execute);
}
#endif
