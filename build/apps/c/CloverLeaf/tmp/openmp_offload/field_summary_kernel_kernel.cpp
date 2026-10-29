// Auto-generated at 2026-10-29 00:13:48.565767 by ops-translator legacy


//  ==================
//  Host stub function
//  ==================
#ifndef OPS_LAZY
void ops_par_loop_field_summary_kernel(
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
void ops_par_loop_field_summary_kernel_execute(ops_kernel_descriptor *desc)
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
    if (!ops_checkpointing_before(args, 11, range, 27)) return;
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timing_realloc(block->instance, 27, "field_summary_kernel");
        block->instance->OPS_kernels[27].count++;
        ops_timers_core(&__c1, &__t1);
    }

#ifdef OPS_DEBUG
    ops_register_args(block->instance, args, "field_summary_kernel");
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
    int xdim0_field_summary_kernel = args[0].dat->size[0];
    int xdim1_field_summary_kernel = args[1].dat->size[0];
    int xdim2_field_summary_kernel = args[2].dat->size[0];
    int xdim3_field_summary_kernel = args[3].dat->size[0];
    int xdim4_field_summary_kernel = args[4].dat->size[0];
    int xdim5_field_summary_kernel = args[5].dat->size[0];

//  =======================================================
//  Set up initial pointers and exchange halos if necessary
//  =======================================================
    int base0 = args[0].dat->base_offset;
    double * __restrict__ volume_p = (double *)(args[0].data_d + base0);

    int base1 = args[1].dat->base_offset;
    double * __restrict__ density0_p = (double *)(args[1].data_d + base1);

    int base2 = args[2].dat->base_offset;
    double * __restrict__ energy0_p = (double *)(args[2].data_d + base2);

    int base3 = args[3].dat->base_offset;
    double * __restrict__ pressure_p = (double *)(args[3].data_d + base3);

    int base4 = args[4].dat->base_offset;
    double * __restrict__ xvel0_p = (double *)(args[4].data_d + base4);

    int base5 = args[5].dat->base_offset;
    double * __restrict__ yvel0_p = (double *)(args[5].data_d + base5);

#ifdef OPS_MPI
    double * __restrict__ p_a6 = (double *)(((ops_reduction)args[6].data)->data + ((ops_reduction)args[6].data)->size * block->index);
#else //OPS_MPI
    double * __restrict__ p_a6 = (double *)((ops_reduction)args[6].data)->data;
#endif //OPS_MPI

#ifdef OPS_MPI
    double * __restrict__ p_a7 = (double *)(((ops_reduction)args[7].data)->data + ((ops_reduction)args[7].data)->size * block->index);
#else //OPS_MPI
    double * __restrict__ p_a7 = (double *)((ops_reduction)args[7].data)->data;
#endif //OPS_MPI

#ifdef OPS_MPI
    double * __restrict__ p_a8 = (double *)(((ops_reduction)args[8].data)->data + ((ops_reduction)args[8].data)->size * block->index);
#else //OPS_MPI
    double * __restrict__ p_a8 = (double *)((ops_reduction)args[8].data)->data;
#endif //OPS_MPI

#ifdef OPS_MPI
    double * __restrict__ p_a9 = (double *)(((ops_reduction)args[9].data)->data + ((ops_reduction)args[9].data)->size * block->index);
#else //OPS_MPI
    double * __restrict__ p_a9 = (double *)((ops_reduction)args[9].data)->data;
#endif //OPS_MPI

#ifdef OPS_MPI
    double * __restrict__ p_a10 = (double *)(((ops_reduction)args[10].data)->data + ((ops_reduction)args[10].data)->size * block->index);
#else //OPS_MPI
    double * __restrict__ p_a10 = (double *)((ops_reduction)args[10].data)->data;
#endif //OPS_MPI


    int consts_bytes = 0;


    reallocConstArrays(block->instance,consts_bytes);
    consts_bytes = 0;

    mvConstArraysToDevice(block->instance,consts_bytes);


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
        block->instance->OPS_kernels[27].mpi_time += __t2 - __t1;
    }

    double p_a6_0 = p_a6[0];
    double p_a7_0 = p_a7[0];
    double p_a8_0 = p_a8[0];
    double p_a9_0 = p_a9[0];
    double p_a10_0 = p_a10[0];
    #pragma omp target teams distribute parallel for collapse(2) reduction(+:p_a6_0) reduction(+:p_a7_0) reduction(+:p_a8_0) reduction(+:p_a9_0) reduction(+:p_a10_0)
    for (int n_y = start1; n_y < end1; n_y++) {
        for (int n_x = start0; n_x < end0; n_x++) {

            const ACC<double> volume(xdim0_field_summary_kernel, volume_p + n_x*1 + n_y * xdim0_field_summary_kernel*1);

            const ACC<double> density0(xdim1_field_summary_kernel, density0_p + n_x*1 + n_y * xdim1_field_summary_kernel*1);

            const ACC<double> energy0(xdim2_field_summary_kernel, energy0_p + n_x*1 + n_y * xdim2_field_summary_kernel*1);

            const ACC<double> pressure(xdim3_field_summary_kernel, pressure_p + n_x*1 + n_y * xdim3_field_summary_kernel*1);

            const ACC<double> xvel0(xdim4_field_summary_kernel, xvel0_p + n_x*1 + n_y * xdim4_field_summary_kernel*1);

            const ACC<double> yvel0(xdim5_field_summary_kernel, yvel0_p + n_x*1 + n_y * xdim5_field_summary_kernel*1);

            double vol[1];
            vol[0] = ZERO_double;

            double mass[1];
            mass[0] = ZERO_double;

            double ie[1];
            ie[0] = ZERO_double;

            double ke[1];
            ke[0] = ZERO_double;

            double press[1];
            press[0] = ZERO_double;

            

  double vsqrd, cell_vol, cell_mass;




  vsqrd = 0.0;
  vsqrd = vsqrd + 0.25 * ( xvel0(0,0) * xvel0(0,0) + yvel0(0,0) * yvel0(0,0));
  vsqrd = vsqrd + 0.25 * ( xvel0(1,0) * xvel0(1,0) + yvel0(1,0) * yvel0(1,0));
  vsqrd = vsqrd + 0.25 * ( xvel0(0,1) * xvel0(0,1) + yvel0(0,1) * yvel0(0,1));
  vsqrd = vsqrd + 0.25 * ( xvel0(1,1) * xvel0(1,1) + yvel0(1,1) * yvel0(1,1));


  cell_vol = volume(0,0);
  cell_mass = cell_vol * density0(0,0);
  *vol = *vol + cell_vol;
  *mass = *mass + cell_mass;
  *ie = *ie + cell_mass * energy0(0,0);
  *ke = *ke + cell_mass * 0.5 * vsqrd;
  *press = *press + cell_vol * pressure(0,0);


            p_a6_0 +=vol[0];
            p_a7_0 +=mass[0];
            p_a8_0 +=ie[0];
            p_a9_0 +=ke[0];
            p_a10_0 +=press[0];
        }
    }

    p_a6[0] = p_a6_0;

    p_a7[0] = p_a7_0;

    p_a8[0] = p_a8_0;

    p_a9[0] = p_a9_0;

    p_a10[0] = p_a10_0;

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c1, &__t1);
        block->instance->OPS_kernels[27].time += __t1 - __t2;
    }
#ifndef OPS_LAZY
    ops_set_dirtybit_device(args, 11);
#endif

    if (block->instance->OPS_diags > 1) {
        //  ====================
        //  Update kernel record
        //  ====================
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[27].mpi_time += __t2 - __t1;
        block->instance->OPS_kernels[27].transfer += ops_compute_transfer(dim, start, end, &arg0);
        block->instance->OPS_kernels[27].transfer += ops_compute_transfer(dim, start, end, &arg1);
        block->instance->OPS_kernels[27].transfer += ops_compute_transfer(dim, start, end, &arg2);
        block->instance->OPS_kernels[27].transfer += ops_compute_transfer(dim, start, end, &arg3);
        block->instance->OPS_kernels[27].transfer += ops_compute_transfer(dim, start, end, &arg4);
        block->instance->OPS_kernels[27].transfer += ops_compute_transfer(dim, start, end, &arg5);
    }
}

#ifdef OPS_LAZY
void ops_par_loop_field_summary_kernel(
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

    create_kerneldesc_and_enque("field_summary_kernel", args, 11, 27, dim, 1, range, block, ops_par_loop_field_summary_kernel_execute);
}
#endif
