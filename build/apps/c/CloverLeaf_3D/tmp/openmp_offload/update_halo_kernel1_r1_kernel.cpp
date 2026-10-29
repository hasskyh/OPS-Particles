// Auto-generated at 2026-10-29 00:13:57.830368 by ops-translator legacy


//  ==================
//  Host stub function
//  ==================
#ifndef OPS_LAZY
void ops_par_loop_update_halo_kernel1_r1(
    const char *name,
    ops_block block,
    int dim,
    int* range,
    ops_arg arg0, 
    ops_arg arg1, 
    ops_arg arg2, 
    ops_arg arg3, 
    ops_arg arg4, 
    ops_arg arg5, 
    ops_arg arg6, 
    ops_arg arg7
)
{
#else
void ops_par_loop_update_halo_kernel1_r1_execute(ops_kernel_descriptor *desc)
{
    ops_block block = desc->block;
    int dim = desc->dim;
    int *range = desc->range;
    ops_arg arg0 = desc->args[0];
    ops_arg arg1 = desc->args[1];
    ops_arg arg2 = desc->args[2];
    ops_arg arg3 = desc->args[3];
    ops_arg arg4 = desc->args[4];
    ops_arg arg5 = desc->args[5];
    ops_arg arg6 = desc->args[6];
    ops_arg arg7 = desc->args[7];
#endif

//  ======
//  Timing
//  ======
    double __t1, __t2, __c1, __c2;

    ops_arg args[8];

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;
    args[3] = arg3;
    args[4] = arg4;
    args[5] = arg5;
    args[6] = arg6;
    args[7] = arg7;

#if defined(CHECKPOINTING) && !defined(OPS_LAZY)
    if (!ops_checkpointing_before(args, 8, range, 63)) return;
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timing_realloc(block->instance, 63, "update_halo_kernel1_r1");
        block->instance->OPS_kernels[63].count++;
        ops_timers_core(&__c1, &__t1);
    }

#ifdef OPS_DEBUG
    ops_register_args(block->instance, args, "update_halo_kernel1_r1");
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
    if (compute_ranges(args, 8, block, range, start, end, arg_idx) < 0) return;
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
    int xdim0_update_halo_kernel1_r1 = args[0].dat->size[0];
    int ydim0_update_halo_kernel1_r1 = args[0].dat->size[1];
    int xdim1_update_halo_kernel1_r1 = args[1].dat->size[0];
    int ydim1_update_halo_kernel1_r1 = args[1].dat->size[1];
    int xdim2_update_halo_kernel1_r1 = args[2].dat->size[0];
    int ydim2_update_halo_kernel1_r1 = args[2].dat->size[1];
    int xdim3_update_halo_kernel1_r1 = args[3].dat->size[0];
    int ydim3_update_halo_kernel1_r1 = args[3].dat->size[1];
    int xdim4_update_halo_kernel1_r1 = args[4].dat->size[0];
    int ydim4_update_halo_kernel1_r1 = args[4].dat->size[1];
    int xdim5_update_halo_kernel1_r1 = args[5].dat->size[0];
    int ydim5_update_halo_kernel1_r1 = args[5].dat->size[1];
    int xdim6_update_halo_kernel1_r1 = args[6].dat->size[0];
    int ydim6_update_halo_kernel1_r1 = args[6].dat->size[1];

//  =======================================================
//  Set up initial pointers and exchange halos if necessary
//  =======================================================
    int base0 = args[0].dat->base_offset;
    double * __restrict__ density0_p = (double *)(args[0].data_d + base0);

    int base1 = args[1].dat->base_offset;
    double * __restrict__ density1_p = (double *)(args[1].data_d + base1);

    int base2 = args[2].dat->base_offset;
    double * __restrict__ energy0_p = (double *)(args[2].data_d + base2);

    int base3 = args[3].dat->base_offset;
    double * __restrict__ energy1_p = (double *)(args[3].data_d + base3);

    int base4 = args[4].dat->base_offset;
    double * __restrict__ pressure_p = (double *)(args[4].data_d + base4);

    int base5 = args[5].dat->base_offset;
    double * __restrict__ viscosity_p = (double *)(args[5].data_d + base5);

    int base6 = args[6].dat->base_offset;
    double * __restrict__ soundspeed_p = (double *)(args[6].data_d + base6);



    int consts_bytes = 0;

    int *arg7h = (int *)args[7].data;
    consts_bytes += ROUND_UP(args[7].dim*sizeof(int));

    reallocConstArrays(block->instance,consts_bytes);
    consts_bytes = 0;

    args[7].data = block->instance->OPS_consts_h + consts_bytes;
    args[7].data_d = block->instance->OPS_consts_d + consts_bytes;
    for (int d = 0; d < args[7].dim; d++) {
        ((int *)args[7].data)[d] = arg7h[d];
    }
    consts_bytes += ROUND_UP(args[7].dim*sizeof(int));

    mvConstArraysToDevice(block->instance,consts_bytes);

    int * __restrict__ fields = (int *)args[7].data_d;

#ifndef OPS_LAZY
//  ==============
//  Halo Exchanges
//  ==============
    ops_H_D_exchanges_device(args, 8);
    ops_halo_exchanges(args, 8,range);
    ops_H_D_exchanges_device(args, 8);
#endif //OPS_LAZY

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[63].mpi_time += __t2 - __t1;
    }

    #pragma omp target teams distribute parallel for collapse(3)
    for (int n_z = start2; n_z < end2; n_z++) {
        for (int n_y = start1; n_y < end1; n_y++) {
            for (int n_x = start0; n_x < end0; n_x++) {

                ACC<double> density0(xdim0_update_halo_kernel1_r1, ydim0_update_halo_kernel1_r1, density0_p + n_x*1 + n_y * xdim0_update_halo_kernel1_r1*1 + n_z * xdim0_update_halo_kernel1_r1 * ydim0_update_halo_kernel1_r1*1);

                ACC<double> density1(xdim1_update_halo_kernel1_r1, ydim1_update_halo_kernel1_r1, density1_p + n_x*1 + n_y * xdim1_update_halo_kernel1_r1*1 + n_z * xdim1_update_halo_kernel1_r1 * ydim1_update_halo_kernel1_r1*1);

                ACC<double> energy0(xdim2_update_halo_kernel1_r1, ydim2_update_halo_kernel1_r1, energy0_p + n_x*1 + n_y * xdim2_update_halo_kernel1_r1*1 + n_z * xdim2_update_halo_kernel1_r1 * ydim2_update_halo_kernel1_r1*1);

                ACC<double> energy1(xdim3_update_halo_kernel1_r1, ydim3_update_halo_kernel1_r1, energy1_p + n_x*1 + n_y * xdim3_update_halo_kernel1_r1*1 + n_z * xdim3_update_halo_kernel1_r1 * ydim3_update_halo_kernel1_r1*1);

                ACC<double> pressure(xdim4_update_halo_kernel1_r1, ydim4_update_halo_kernel1_r1, pressure_p + n_x*1 + n_y * xdim4_update_halo_kernel1_r1*1 + n_z * xdim4_update_halo_kernel1_r1 * ydim4_update_halo_kernel1_r1*1);

                ACC<double> viscosity(xdim5_update_halo_kernel1_r1, ydim5_update_halo_kernel1_r1, viscosity_p + n_x*1 + n_y * xdim5_update_halo_kernel1_r1*1 + n_z * xdim5_update_halo_kernel1_r1 * ydim5_update_halo_kernel1_r1*1);

                ACC<double> soundspeed(xdim6_update_halo_kernel1_r1, ydim6_update_halo_kernel1_r1, soundspeed_p + n_x*1 + n_y * xdim6_update_halo_kernel1_r1*1 + n_z * xdim6_update_halo_kernel1_r1 * ydim6_update_halo_kernel1_r1*1);

                
  if(fields[FIELD_DENSITY0] == 1) density0(0,0,0) = density0(-1,0,0);
  if(fields[FIELD_DENSITY1] == 1) density1(0,0,0) = density1(-1,0,0);
  if(fields[FIELD_ENERGY0] == 1) energy0(0,0,0) = energy0(-1,0,0);
  if(fields[FIELD_ENERGY1] == 1) energy1(0,0,0) = energy1(-1,0,0);
  if(fields[FIELD_PRESSURE] == 1) pressure(0,0,0) = pressure(-1,0,0);
  if(fields[FIELD_VISCOSITY] == 1) viscosity(0,0,0) = viscosity(-1,0,0);
  if(fields[FIELD_SOUNDSPEED] == 1) soundspeed(0,0,0) = soundspeed(-1,0,0);


            }
        }
    }

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c1, &__t1);
        block->instance->OPS_kernels[63].time += __t1 - __t2;
    }
#ifndef OPS_LAZY
    ops_set_dirtybit_device(args, 8);
    ops_set_halo_dirtybit3(&args[0], range);
    ops_set_halo_dirtybit3(&args[1], range);
    ops_set_halo_dirtybit3(&args[2], range);
    ops_set_halo_dirtybit3(&args[3], range);
    ops_set_halo_dirtybit3(&args[4], range);
    ops_set_halo_dirtybit3(&args[5], range);
    ops_set_halo_dirtybit3(&args[6], range);
#endif

    if (block->instance->OPS_diags > 1) {
        //  ====================
        //  Update kernel record
        //  ====================
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[63].mpi_time += __t2 - __t1;
        block->instance->OPS_kernels[63].transfer += ops_compute_transfer(dim, start, end, &arg0);
        block->instance->OPS_kernels[63].transfer += ops_compute_transfer(dim, start, end, &arg1);
        block->instance->OPS_kernels[63].transfer += ops_compute_transfer(dim, start, end, &arg2);
        block->instance->OPS_kernels[63].transfer += ops_compute_transfer(dim, start, end, &arg3);
        block->instance->OPS_kernels[63].transfer += ops_compute_transfer(dim, start, end, &arg4);
        block->instance->OPS_kernels[63].transfer += ops_compute_transfer(dim, start, end, &arg5);
        block->instance->OPS_kernels[63].transfer += ops_compute_transfer(dim, start, end, &arg6);
    }
}

#ifdef OPS_LAZY
void ops_par_loop_update_halo_kernel1_r1(
    const char *name,
    ops_block block,
    int dim,
    int* range,
    ops_arg arg0,
    ops_arg arg1,
    ops_arg arg2,
    ops_arg arg3,
    ops_arg arg4,
    ops_arg arg5,
    ops_arg arg6,
    ops_arg arg7
)
{
    ops_arg args[8];

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;
    args[3] = arg3;
    args[4] = arg4;
    args[5] = arg5;
    args[6] = arg6;
    args[7] = arg7;

    create_kerneldesc_and_enque("update_halo_kernel1_r1", args, 8, 63, dim, 1, range, block, ops_par_loop_update_halo_kernel1_r1_execute);
}
#endif
