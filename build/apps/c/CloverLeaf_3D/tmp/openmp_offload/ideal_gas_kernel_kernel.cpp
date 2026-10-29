// Auto-generated at 2026-10-29 00:13:57.508225 by ops-translator legacy


//  ==================
//  Host stub function
//  ==================
#ifndef OPS_LAZY
void ops_par_loop_ideal_gas_kernel(
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
void ops_par_loop_ideal_gas_kernel_execute(ops_kernel_descriptor *desc)
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
    if (!ops_checkpointing_before(args, 4, range, 42)) return;
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timing_realloc(block->instance, 42, "ideal_gas_kernel");
        block->instance->OPS_kernels[42].count++;
        ops_timers_core(&__c1, &__t1);
    }

#ifdef OPS_DEBUG
    ops_register_args(block->instance, args, "ideal_gas_kernel");
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
    int xdim0_ideal_gas_kernel = args[0].dat->size[0];
    int ydim0_ideal_gas_kernel = args[0].dat->size[1];
    int xdim1_ideal_gas_kernel = args[1].dat->size[0];
    int ydim1_ideal_gas_kernel = args[1].dat->size[1];
    int xdim2_ideal_gas_kernel = args[2].dat->size[0];
    int ydim2_ideal_gas_kernel = args[2].dat->size[1];
    int xdim3_ideal_gas_kernel = args[3].dat->size[0];
    int ydim3_ideal_gas_kernel = args[3].dat->size[1];

//  =======================================================
//  Set up initial pointers and exchange halos if necessary
//  =======================================================
    int base0 = args[0].dat->base_offset;
    double * __restrict__ density_p = (double *)(args[0].data_d + base0);

    int base1 = args[1].dat->base_offset;
    double * __restrict__ energy_p = (double *)(args[1].data_d + base1);

    int base2 = args[2].dat->base_offset;
    double * __restrict__ pressure_p = (double *)(args[2].data_d + base2);

    int base3 = args[3].dat->base_offset;
    double * __restrict__ soundspeed_p = (double *)(args[3].data_d + base3);

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
        block->instance->OPS_kernels[42].mpi_time += __t2 - __t1;
    }

    #pragma omp target teams distribute parallel for collapse(3)
    for (int n_z = start2; n_z < end2; n_z++) {
        for (int n_y = start1; n_y < end1; n_y++) {
            for (int n_x = start0; n_x < end0; n_x++) {

                const ACC<double> density(xdim0_ideal_gas_kernel, ydim0_ideal_gas_kernel, density_p + n_x*1 + n_y * xdim0_ideal_gas_kernel*1 + n_z * xdim0_ideal_gas_kernel * ydim0_ideal_gas_kernel*1);

                const ACC<double> energy(xdim1_ideal_gas_kernel, ydim1_ideal_gas_kernel, energy_p + n_x*1 + n_y * xdim1_ideal_gas_kernel*1 + n_z * xdim1_ideal_gas_kernel * ydim1_ideal_gas_kernel*1);

                ACC<double> pressure(xdim2_ideal_gas_kernel, ydim2_ideal_gas_kernel, pressure_p + n_x*1 + n_y * xdim2_ideal_gas_kernel*1 + n_z * xdim2_ideal_gas_kernel * ydim2_ideal_gas_kernel*1);

                ACC<double> soundspeed(xdim3_ideal_gas_kernel, ydim3_ideal_gas_kernel, soundspeed_p + n_x*1 + n_y * xdim3_ideal_gas_kernel*1 + n_z * xdim3_ideal_gas_kernel * ydim3_ideal_gas_kernel*1);

                

  double sound_speed_squared, v, pressurebyenergy, pressurebyvolume;

  v = 1.0 / density(0,0,0);
  pressure(0,0,0) = (1.4 - 1.0) * density(0,0,0) * energy(0,0,0);

  pressurebyenergy = (1.4 - 1.0) * density(0,0,0);
  pressurebyvolume = -1.0*density(0,0,0) * pressure(0,0,0);
  sound_speed_squared = v*v*(pressure(0,0,0) * pressurebyenergy-pressurebyvolume);
  soundspeed(0,0,0) = sqrt(sound_speed_squared);

            }
        }
    }

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c1, &__t1);
        block->instance->OPS_kernels[42].time += __t1 - __t2;
    }
#ifndef OPS_LAZY
    ops_set_dirtybit_device(args, 4);
    ops_set_halo_dirtybit3(&args[2], range);
    ops_set_halo_dirtybit3(&args[3], range);
#endif

    if (block->instance->OPS_diags > 1) {
        //  ====================
        //  Update kernel record
        //  ====================
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[42].mpi_time += __t2 - __t1;
        block->instance->OPS_kernels[42].transfer += ops_compute_transfer(dim, start, end, &arg0);
        block->instance->OPS_kernels[42].transfer += ops_compute_transfer(dim, start, end, &arg1);
        block->instance->OPS_kernels[42].transfer += ops_compute_transfer(dim, start, end, &arg2);
        block->instance->OPS_kernels[42].transfer += ops_compute_transfer(dim, start, end, &arg3);
    }
}

#ifdef OPS_LAZY
void ops_par_loop_ideal_gas_kernel(
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

    create_kerneldesc_and_enque("ideal_gas_kernel", args, 4, 42, dim, 1, range, block, ops_par_loop_ideal_gas_kernel_execute);
}
#endif
