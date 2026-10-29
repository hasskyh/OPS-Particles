// Auto-generated at 2026-10-29 00:13:48.530757 by ops-translator legacy


//  ==================
//  Host stub function
//  ==================
#ifndef OPS_LAZY
void ops_par_loop_calc_dt_kernel(
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
    ops_arg arg10
)
{
#else
void ops_par_loop_calc_dt_kernel_execute(ops_kernel_descriptor *desc)
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
#endif

//  ======
//  Timing
//  ======
    double __t1, __t2, __c1, __c2;

    ops_arg args[11];

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

#if defined(CHECKPOINTING) && !defined(OPS_LAZY)
    if (!ops_checkpointing_before(args, 11, range, 23)) return;
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timing_realloc(block->instance, 23, "calc_dt_kernel");
        block->instance->OPS_kernels[23].count++;
        ops_timers_core(&__c1, &__t1);
    }

#ifdef OPS_DEBUG
    ops_register_args(block->instance, args, "calc_dt_kernel");
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
    if (compute_ranges(args, 11, block, range, start, end, arg_idx) < 0) return;
#endif

    int start0 = start[0];
    int end0 = end[0];
    int start1 = start[1];
    int end1 = end[1];

    //  =====================================================
    //  Initialize global variable with the dimension of dats
    //  =====================================================
    int xdim0_calc_dt_kernel = args[0].dat->size[0];
    int xdim1_calc_dt_kernel = args[1].dat->size[0];
    int xdim2_calc_dt_kernel = args[2].dat->size[0];
    int xdim3_calc_dt_kernel = args[3].dat->size[0];
    int xdim4_calc_dt_kernel = args[4].dat->size[0];
    int xdim5_calc_dt_kernel = args[5].dat->size[0];
    int xdim6_calc_dt_kernel = args[6].dat->size[0];
    int xdim7_calc_dt_kernel = args[7].dat->size[0];
    int xdim8_calc_dt_kernel = args[8].dat->size[0];
    int xdim9_calc_dt_kernel = args[9].dat->size[0];
    int xdim10_calc_dt_kernel = args[10].dat->size[0];

//  =======================================================
//  Set up initial pointers and exchange halos if necessary
//  =======================================================
    int base0 = args[0].dat->base_offset;
    double * __restrict__ celldx_p = (double *)(args[0].data_d + base0);

    int base1 = args[1].dat->base_offset;
    double * __restrict__ celldy_p = (double *)(args[1].data_d + base1);

    int base2 = args[2].dat->base_offset;
    double * __restrict__ soundspeed_p = (double *)(args[2].data_d + base2);

    int base3 = args[3].dat->base_offset;
    double * __restrict__ viscosity_p = (double *)(args[3].data_d + base3);

    int base4 = args[4].dat->base_offset;
    double * __restrict__ density0_p = (double *)(args[4].data_d + base4);

    int base5 = args[5].dat->base_offset;
    double * __restrict__ xvel0_p = (double *)(args[5].data_d + base5);

    int base6 = args[6].dat->base_offset;
    double * __restrict__ xarea_p = (double *)(args[6].data_d + base6);

    int base7 = args[7].dat->base_offset;
    double * __restrict__ volume_p = (double *)(args[7].data_d + base7);

    int base8 = args[8].dat->base_offset;
    double * __restrict__ yvel0_p = (double *)(args[8].data_d + base8);

    int base9 = args[9].dat->base_offset;
    double * __restrict__ yarea_p = (double *)(args[9].data_d + base9);

    int base10 = args[10].dat->base_offset;
    double * __restrict__ dt_min_p = (double *)(args[10].data_d + base10);

#ifndef OPS_LAZY
//  ==============
//  Halo Exchanges
//  ==============
    ops_H_D_exchanges_device(args, 11);
    ops_halo_exchanges(args, 11,range);
    ops_H_D_exchanges_device(args, 11);
#endif //OPS_LAZY

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[23].mpi_time += __t2 - __t1;
    }

    #pragma omp target teams distribute parallel for collapse(2)
    for (int n_y = start1; n_y < end1; n_y++) {
        for (int n_x = start0; n_x < end0; n_x++) {

            const ACC<double> celldx(xdim0_calc_dt_kernel, celldx_p + n_x*1 + n_y * xdim0_calc_dt_kernel*0);

            const ACC<double> celldy(xdim1_calc_dt_kernel, celldy_p + n_x*0 + n_y * xdim1_calc_dt_kernel*1);

            const ACC<double> soundspeed(xdim2_calc_dt_kernel, soundspeed_p + n_x*1 + n_y * xdim2_calc_dt_kernel*1);

            const ACC<double> viscosity(xdim3_calc_dt_kernel, viscosity_p + n_x*1 + n_y * xdim3_calc_dt_kernel*1);

            const ACC<double> density0(xdim4_calc_dt_kernel, density0_p + n_x*1 + n_y * xdim4_calc_dt_kernel*1);

            const ACC<double> xvel0(xdim5_calc_dt_kernel, xvel0_p + n_x*1 + n_y * xdim5_calc_dt_kernel*1);

            const ACC<double> xarea(xdim6_calc_dt_kernel, xarea_p + n_x*1 + n_y * xdim6_calc_dt_kernel*1);

            const ACC<double> volume(xdim7_calc_dt_kernel, volume_p + n_x*1 + n_y * xdim7_calc_dt_kernel*1);

            const ACC<double> yvel0(xdim8_calc_dt_kernel, yvel0_p + n_x*1 + n_y * xdim8_calc_dt_kernel*1);

            const ACC<double> yarea(xdim9_calc_dt_kernel, yarea_p + n_x*1 + n_y * xdim9_calc_dt_kernel*1);

            ACC<double> dt_min(xdim10_calc_dt_kernel, dt_min_p + n_x*1 + n_y * xdim10_calc_dt_kernel*1);

            

  double div, dsx, dsy, dtut, dtvt, dtct, dtdivt, cc, dv1, dv2;

  dsx = celldx(0,0);
  dsy = celldy(0,0);

  cc = soundspeed(0,0) * soundspeed(0,0);
  cc = cc + 2.0 * viscosity(0,0)/density0(0,0);
  cc = MAX(sqrt(cc),g_small);

  dtct = dtc_safe * MIN(dsx,dsy)/cc;

  div=0.0;



  dv1 = (xvel0(0,0) + xvel0(0,1)) * xarea(0,0);
  dv2 = (xvel0(1,0) + xvel0(1,1)) * xarea(1,0);

  div = div + dv2 - dv1;

  dtut = dtu_safe * 2.0 * volume(0,0)/MAX(MAX(fabs(dv1), fabs(dv2)), g_small * volume(0,0));

  dv1 = (yvel0(0,0) + yvel0(1,0)) * yarea(0,0);
  dv2 = (yvel0(0,1) + yvel0(1,1)) * yarea(0,1);

  div = div + dv2 - dv1;

  dtvt = dtv_safe * 2.0 * volume(0,0)/MAX(MAX(fabs(dv1),fabs(dv2)), g_small * volume(0,0));

  div = div/(2.0 * volume(0,0));

  if(div < -g_small)
    dtdivt = dtdiv_safe * (-1.0/div);
  else
    dtdivt = g_big;


  dt_min(0,0) = MIN(MIN(dtct, dtut), MIN(dtvt, dtdivt));



        }
    }

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c1, &__t1);
        block->instance->OPS_kernels[23].time += __t1 - __t2;
    }
#ifndef OPS_LAZY
    ops_set_dirtybit_device(args, 11);
    ops_set_halo_dirtybit3(&args[10], range);
#endif

    if (block->instance->OPS_diags > 1) {
        //  ====================
        //  Update kernel record
        //  ====================
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[23].mpi_time += __t2 - __t1;
        block->instance->OPS_kernels[23].transfer += ops_compute_transfer(dim, start, end, &arg0);
        block->instance->OPS_kernels[23].transfer += ops_compute_transfer(dim, start, end, &arg1);
        block->instance->OPS_kernels[23].transfer += ops_compute_transfer(dim, start, end, &arg2);
        block->instance->OPS_kernels[23].transfer += ops_compute_transfer(dim, start, end, &arg3);
        block->instance->OPS_kernels[23].transfer += ops_compute_transfer(dim, start, end, &arg4);
        block->instance->OPS_kernels[23].transfer += ops_compute_transfer(dim, start, end, &arg5);
        block->instance->OPS_kernels[23].transfer += ops_compute_transfer(dim, start, end, &arg6);
        block->instance->OPS_kernels[23].transfer += ops_compute_transfer(dim, start, end, &arg7);
        block->instance->OPS_kernels[23].transfer += ops_compute_transfer(dim, start, end, &arg8);
        block->instance->OPS_kernels[23].transfer += ops_compute_transfer(dim, start, end, &arg9);
        block->instance->OPS_kernels[23].transfer += ops_compute_transfer(dim, start, end, &arg10);
    }
}

#ifdef OPS_LAZY
void ops_par_loop_calc_dt_kernel(
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
    ops_arg arg10
)
{
    ops_arg args[11];

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

    create_kerneldesc_and_enque("calc_dt_kernel", args, 11, 23, dim, 1, range, block, ops_par_loop_calc_dt_kernel_execute);
}
#endif
