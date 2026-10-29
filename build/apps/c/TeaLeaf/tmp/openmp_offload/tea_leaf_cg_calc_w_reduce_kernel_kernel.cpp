// Auto-generated at 2026-10-29 00:14:08.671936 by ops-translator legacy


//  ==================
//  Host stub function
//  ==================
#ifndef OPS_LAZY
void ops_par_loop_tea_leaf_cg_calc_w_reduce_kernel(
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
    ops_arg arg6
)
{
#else
void ops_par_loop_tea_leaf_cg_calc_w_reduce_kernel_execute(ops_kernel_descriptor *desc)
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
#endif

//  ======
//  Timing
//  ======
    double __t1, __t2, __c1, __c2;

    ops_arg args[7];

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;
    args[3] = arg3;
    args[4] = arg4;
    args[5] = arg5;
    args[6] = arg6;

#if defined(CHECKPOINTING) && !defined(OPS_LAZY)
    if (!ops_checkpointing_before(args, 7, range, 16)) return;
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timing_realloc(block->instance, 16, "tea_leaf_cg_calc_w_reduce_kernel");
        block->instance->OPS_kernels[16].count++;
        ops_timers_core(&__c1, &__t1);
    }

#ifdef OPS_DEBUG
    ops_register_args(block->instance, args, "tea_leaf_cg_calc_w_reduce_kernel");
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
    if (compute_ranges(args, 7, block, range, start, end, arg_idx) < 0) return;
#endif

    int start0 = start[0];
    int end0 = end[0];
    int start1 = start[1];
    int end1 = end[1];

    //  =====================================================
    //  Initialize global variable with the dimension of dats
    //  =====================================================
    int xdim0_tea_leaf_cg_calc_w_reduce_kernel = args[0].dat->size[0];
    int xdim1_tea_leaf_cg_calc_w_reduce_kernel = args[1].dat->size[0];
    int xdim2_tea_leaf_cg_calc_w_reduce_kernel = args[2].dat->size[0];
    int xdim3_tea_leaf_cg_calc_w_reduce_kernel = args[3].dat->size[0];

//  =======================================================
//  Set up initial pointers and exchange halos if necessary
//  =======================================================
    int base0 = args[0].dat->base_offset;
    double * __restrict__ w_p = (double *)(args[0].data_d + base0);

    int base1 = args[1].dat->base_offset;
    double * __restrict__ Kx_p = (double *)(args[1].data_d + base1);

    int base2 = args[2].dat->base_offset;
    double * __restrict__ Ky_p = (double *)(args[2].data_d + base2);

    int base3 = args[3].dat->base_offset;
    double * __restrict__ p_p = (double *)(args[3].data_d + base3);



#ifdef OPS_MPI
    double * __restrict__ p_a6 = (double *)(((ops_reduction)args[6].data)->data + ((ops_reduction)args[6].data)->size * block->index);
#else //OPS_MPI
    double * __restrict__ p_a6 = (double *)((ops_reduction)args[6].data)->data;
#endif //OPS_MPI


    int consts_bytes = 0;

    double *arg4h = (double *)args[4].data;
    consts_bytes += ROUND_UP(args[4].dim*sizeof(double));
    double *arg5h = (double *)args[5].data;
    consts_bytes += ROUND_UP(args[5].dim*sizeof(double));

    reallocConstArrays(block->instance,consts_bytes);
    consts_bytes = 0;

    args[4].data = block->instance->OPS_consts_h + consts_bytes;
    args[4].data_d = block->instance->OPS_consts_d + consts_bytes;
    for (int d = 0; d < args[4].dim; d++) {
        ((double *)args[4].data)[d] = arg4h[d];
    }
    consts_bytes += ROUND_UP(args[4].dim*sizeof(double));

    args[5].data = block->instance->OPS_consts_h + consts_bytes;
    args[5].data_d = block->instance->OPS_consts_d + consts_bytes;
    for (int d = 0; d < args[5].dim; d++) {
        ((double *)args[5].data)[d] = arg5h[d];
    }
    consts_bytes += ROUND_UP(args[5].dim*sizeof(double));

    mvConstArraysToDevice(block->instance,consts_bytes);

    double * __restrict__ rx = (double *)args[4].data_d;
    double * __restrict__ ry = (double *)args[5].data_d;

#ifndef OPS_LAZY
//  ==============
//  Halo Exchanges
//  ==============
    ops_H_D_exchanges_device(args, 7);
    ops_halo_exchanges(args, 7,range);
    ops_H_D_exchanges_device(args, 7);
#endif //OPS_LAZY

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[16].mpi_time += __t2 - __t1;
    }

    double p_a6_0 = p_a6[0];
    #pragma omp target teams distribute parallel for collapse(2) reduction(+:p_a6_0)
    for (int n_y = start1; n_y < end1; n_y++) {
        for (int n_x = start0; n_x < end0; n_x++) {

            ACC<double> w(xdim0_tea_leaf_cg_calc_w_reduce_kernel, w_p + n_x*1 + n_y * xdim0_tea_leaf_cg_calc_w_reduce_kernel*1);

            const ACC<double> Kx(xdim1_tea_leaf_cg_calc_w_reduce_kernel, Kx_p + n_x*1 + n_y * xdim1_tea_leaf_cg_calc_w_reduce_kernel*1);

            const ACC<double> Ky(xdim2_tea_leaf_cg_calc_w_reduce_kernel, Ky_p + n_x*1 + n_y * xdim2_tea_leaf_cg_calc_w_reduce_kernel*1);

            const ACC<double> p(xdim3_tea_leaf_cg_calc_w_reduce_kernel, p_p + n_x*1 + n_y * xdim3_tea_leaf_cg_calc_w_reduce_kernel*1);

            double pw[1];
            pw[0] = ZERO_double;

            
  w(0,0) = (1.0
                + (*ry)*(Ky(0,1) + Ky(0,0))
                + (*rx)*(Kx(1,0) + Kx(0,0)))*p(0,0)
                - (*ry)*(Ky(0,1)*p(0,1) + Ky(0,0)*p(0,-1))
                - (*rx)*(Kx(1,0)*p(1,0) + Kx(0,0)*p(-1,0));
  *pw = *pw + w(0,0)*p(0,0);

            p_a6_0 +=pw[0];
        }
    }

    p_a6[0] = p_a6_0;

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c1, &__t1);
        block->instance->OPS_kernels[16].time += __t1 - __t2;
    }
#ifndef OPS_LAZY
    ops_set_dirtybit_device(args, 7);
    ops_set_halo_dirtybit3(&args[0], range);
#endif

    if (block->instance->OPS_diags > 1) {
        //  ====================
        //  Update kernel record
        //  ====================
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[16].mpi_time += __t2 - __t1;
        block->instance->OPS_kernels[16].transfer += ops_compute_transfer(dim, start, end, &arg0);
        block->instance->OPS_kernels[16].transfer += ops_compute_transfer(dim, start, end, &arg1);
        block->instance->OPS_kernels[16].transfer += ops_compute_transfer(dim, start, end, &arg2);
        block->instance->OPS_kernels[16].transfer += ops_compute_transfer(dim, start, end, &arg3);
    }
}

#ifdef OPS_LAZY
void ops_par_loop_tea_leaf_cg_calc_w_reduce_kernel(
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
    ops_arg arg6
)
{
    ops_arg args[7];

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;
    args[3] = arg3;
    args[4] = arg4;
    args[5] = arg5;
    args[6] = arg6;

    create_kerneldesc_and_enque("tea_leaf_cg_calc_w_reduce_kernel", args, 7, 16, dim, 1, range, block, ops_par_loop_tea_leaf_cg_calc_w_reduce_kernel_execute);
}
#endif
