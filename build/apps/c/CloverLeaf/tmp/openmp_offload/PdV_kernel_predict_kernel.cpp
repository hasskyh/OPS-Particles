// Auto-generated at 2026-10-29 00:13:48.399598 by ops-translator legacy


//  ==================
//  Host stub function
//  ==================
#ifndef OPS_LAZY
void ops_par_loop_PdV_kernel_predict(
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
    ops_arg arg7, 
    ops_arg arg8, 
    ops_arg arg9, 
    ops_arg arg10, 
    ops_arg arg11
)
{
#else
void ops_par_loop_PdV_kernel_predict_execute(ops_kernel_descriptor *desc)
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
    ops_arg arg8 = desc->args[8];
    ops_arg arg9 = desc->args[9];
    ops_arg arg10 = desc->args[10];
    ops_arg arg11 = desc->args[11];
#endif

//  ======
//  Timing
//  ======
    double __t1, __t2, __c1, __c2;

    ops_arg args[12];

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;
    args[3] = arg3;
    args[4] = arg4;
    args[5] = arg5;
    args[6] = arg6;
    args[7] = arg7;
    args[8] = arg8;
    args[9] = arg9;
    args[10] = arg10;
    args[11] = arg11;

#if defined(CHECKPOINTING) && !defined(OPS_LAZY)
    if (!ops_checkpointing_before(args, 12, range, 0)) return;
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timing_realloc(block->instance, 0, "PdV_kernel_predict");
        block->instance->OPS_kernels[0].count++;
        ops_timers_core(&__c1, &__t1);
    }

#ifdef OPS_DEBUG
    ops_register_args(block->instance, args, "PdV_kernel_predict");
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
    if (compute_ranges(args, 12, block, range, start, end, arg_idx) < 0) return;
#endif

    int start0 = start[0];
    int end0 = end[0];
    int start1 = start[1];
    int end1 = end[1];

    //  =====================================================
    //  Initialize global variable with the dimension of dats
    //  =====================================================
    int xdim0_PdV_kernel_predict = args[0].dat->size[0];
    int xdim1_PdV_kernel_predict = args[1].dat->size[0];
    int xdim2_PdV_kernel_predict = args[2].dat->size[0];
    int xdim3_PdV_kernel_predict = args[3].dat->size[0];
    int xdim4_PdV_kernel_predict = args[4].dat->size[0];
    int xdim5_PdV_kernel_predict = args[5].dat->size[0];
    int xdim6_PdV_kernel_predict = args[6].dat->size[0];
    int xdim7_PdV_kernel_predict = args[7].dat->size[0];
    int xdim8_PdV_kernel_predict = args[8].dat->size[0];
    int xdim9_PdV_kernel_predict = args[9].dat->size[0];
    int xdim10_PdV_kernel_predict = args[10].dat->size[0];
    int xdim11_PdV_kernel_predict = args[11].dat->size[0];

//  =======================================================
//  Set up initial pointers and exchange halos if necessary
//  =======================================================
    int base0 = args[0].dat->base_offset;
    double * __restrict__ xarea_p = (double *)(args[0].data_d + base0);

    int base1 = args[1].dat->base_offset;
    double * __restrict__ xvel0_p = (double *)(args[1].data_d + base1);

    int base2 = args[2].dat->base_offset;
    double * __restrict__ yarea_p = (double *)(args[2].data_d + base2);

    int base3 = args[3].dat->base_offset;
    double * __restrict__ yvel0_p = (double *)(args[3].data_d + base3);

    int base4 = args[4].dat->base_offset;
    double * __restrict__ volume_change_p = (double *)(args[4].data_d + base4);

    int base5 = args[5].dat->base_offset;
    double * __restrict__ volume_p = (double *)(args[5].data_d + base5);

    int base6 = args[6].dat->base_offset;
    double * __restrict__ pressure_p = (double *)(args[6].data_d + base6);

    int base7 = args[7].dat->base_offset;
    double * __restrict__ density0_p = (double *)(args[7].data_d + base7);

    int base8 = args[8].dat->base_offset;
    double * __restrict__ density1_p = (double *)(args[8].data_d + base8);

    int base9 = args[9].dat->base_offset;
    double * __restrict__ viscosity_p = (double *)(args[9].data_d + base9);

    int base10 = args[10].dat->base_offset;
    double * __restrict__ energy0_p = (double *)(args[10].data_d + base10);

    int base11 = args[11].dat->base_offset;
    double * __restrict__ energy1_p = (double *)(args[11].data_d + base11);

#ifndef OPS_LAZY
//  ==============
//  Halo Exchanges
//  ==============
    ops_H_D_exchanges_device(args, 12);
    ops_halo_exchanges(args, 12,range);
    ops_H_D_exchanges_device(args, 12);
#endif //OPS_LAZY

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[0].mpi_time += __t2 - __t1;
    }

    #pragma omp target teams distribute parallel for collapse(2)
    for (int n_y = start1; n_y < end1; n_y++) {
        for (int n_x = start0; n_x < end0; n_x++) {

            const ACC<double> xarea(xdim0_PdV_kernel_predict, xarea_p + n_x*1 + n_y * xdim0_PdV_kernel_predict*1);

            const ACC<double> xvel0(xdim1_PdV_kernel_predict, xvel0_p + n_x*1 + n_y * xdim1_PdV_kernel_predict*1);

            const ACC<double> yarea(xdim2_PdV_kernel_predict, yarea_p + n_x*1 + n_y * xdim2_PdV_kernel_predict*1);

            const ACC<double> yvel0(xdim3_PdV_kernel_predict, yvel0_p + n_x*1 + n_y * xdim3_PdV_kernel_predict*1);

            ACC<double> volume_change(xdim4_PdV_kernel_predict, volume_change_p + n_x*1 + n_y * xdim4_PdV_kernel_predict*1);

            const ACC<double> volume(xdim5_PdV_kernel_predict, volume_p + n_x*1 + n_y * xdim5_PdV_kernel_predict*1);

            const ACC<double> pressure(xdim6_PdV_kernel_predict, pressure_p + n_x*1 + n_y * xdim6_PdV_kernel_predict*1);

            const ACC<double> density0(xdim7_PdV_kernel_predict, density0_p + n_x*1 + n_y * xdim7_PdV_kernel_predict*1);

            ACC<double> density1(xdim8_PdV_kernel_predict, density1_p + n_x*1 + n_y * xdim8_PdV_kernel_predict*1);

            const ACC<double> viscosity(xdim9_PdV_kernel_predict, viscosity_p + n_x*1 + n_y * xdim9_PdV_kernel_predict*1);

            const ACC<double> energy0(xdim10_PdV_kernel_predict, energy0_p + n_x*1 + n_y * xdim10_PdV_kernel_predict*1);

            ACC<double> energy1(xdim11_PdV_kernel_predict, energy1_p + n_x*1 + n_y * xdim11_PdV_kernel_predict*1);

            



  double recip_volume, energy_change;
  double right_flux, left_flux, top_flux, bottom_flux, total_flux;

  left_flux = ( xarea(0,0) * ( xvel0(0,0) + xvel0(0,1) +
                                xvel0(0,0) + xvel0(0,1) ) ) * 0.25 * dt * 0.5;
  right_flux = ( xarea(1,0) * ( xvel0(1,0) + xvel0(1,1) +
                                 xvel0(1,0) + xvel0(1,1) ) ) * 0.25 * dt * 0.5;

  bottom_flux = ( yarea(0,0) * ( yvel0(0,0) + yvel0(1,0) +
                                  yvel0(0,0) + yvel0(1,0) ) ) * 0.25* dt * 0.5;
  top_flux = ( yarea(0,1) * ( yvel0(0,1) + yvel0(1,1) +
                               yvel0(0,1) + yvel0(1,1) ) ) * 0.25 * dt * 0.5;

  total_flux = right_flux - left_flux + top_flux - bottom_flux;

  volume_change(0,0) = (volume(0,0))/(volume(0,0) + total_flux);





  recip_volume = 1.0/volume(0,0);

  energy_change = ( pressure(0,0)/density0(0,0) +
                    viscosity(0,0)/density0(0,0) ) * total_flux * recip_volume;
  energy1(0,0) = energy0(0,0) - energy_change;
  density1(0,0) = density0(0,0) * volume_change(0,0);


        }
    }

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c1, &__t1);
        block->instance->OPS_kernels[0].time += __t1 - __t2;
    }
#ifndef OPS_LAZY
    ops_set_dirtybit_device(args, 12);
    ops_set_halo_dirtybit3(&args[4], range);
    ops_set_halo_dirtybit3(&args[8], range);
    ops_set_halo_dirtybit3(&args[11], range);
#endif

    if (block->instance->OPS_diags > 1) {
        //  ====================
        //  Update kernel record
        //  ====================
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[0].mpi_time += __t2 - __t1;
        block->instance->OPS_kernels[0].transfer += ops_compute_transfer(dim, start, end, &arg0);
        block->instance->OPS_kernels[0].transfer += ops_compute_transfer(dim, start, end, &arg1);
        block->instance->OPS_kernels[0].transfer += ops_compute_transfer(dim, start, end, &arg2);
        block->instance->OPS_kernels[0].transfer += ops_compute_transfer(dim, start, end, &arg3);
        block->instance->OPS_kernels[0].transfer += ops_compute_transfer(dim, start, end, &arg4);
        block->instance->OPS_kernels[0].transfer += ops_compute_transfer(dim, start, end, &arg5);
        block->instance->OPS_kernels[0].transfer += ops_compute_transfer(dim, start, end, &arg6);
        block->instance->OPS_kernels[0].transfer += ops_compute_transfer(dim, start, end, &arg7);
        block->instance->OPS_kernels[0].transfer += ops_compute_transfer(dim, start, end, &arg8);
        block->instance->OPS_kernels[0].transfer += ops_compute_transfer(dim, start, end, &arg9);
        block->instance->OPS_kernels[0].transfer += ops_compute_transfer(dim, start, end, &arg10);
        block->instance->OPS_kernels[0].transfer += ops_compute_transfer(dim, start, end, &arg11);
    }
}

#ifdef OPS_LAZY
void ops_par_loop_PdV_kernel_predict(
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
    ops_arg arg7,
    ops_arg arg8,
    ops_arg arg9,
    ops_arg arg10,
    ops_arg arg11
)
{
    ops_arg args[12];

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;
    args[3] = arg3;
    args[4] = arg4;
    args[5] = arg5;
    args[6] = arg6;
    args[7] = arg7;
    args[8] = arg8;
    args[9] = arg9;
    args[10] = arg10;
    args[11] = arg11;

    create_kerneldesc_and_enque("PdV_kernel_predict", args, 12, 0, dim, 1, range, block, ops_par_loop_PdV_kernel_predict_execute);
}
#endif
