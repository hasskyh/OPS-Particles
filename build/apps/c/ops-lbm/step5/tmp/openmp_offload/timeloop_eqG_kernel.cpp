// Auto-generated at 2026-10-29 00:14:12.306530 by ops-translator legacy


//  ==================
//  Host stub function
//  ==================
#ifndef OPS_LAZY
void ops_par_loop_timeloop_eqG(
    const char *name,
    ops_block block,
    int dim,
    int* range,
    ops_arg arg0, 
    ops_arg arg1, 
    ops_arg arg2, 
    ops_arg arg3, 
    ops_arg arg4
)
{
#else
void ops_par_loop_timeloop_eqG_execute(ops_kernel_descriptor *desc)
{
    ops_block block = desc->block;
    int dim = desc->dim;
    int *range = desc->range;
    ops_arg arg0 = desc->args[0];
    ops_arg arg1 = desc->args[1];
    ops_arg arg2 = desc->args[2];
    ops_arg arg3 = desc->args[3];
    ops_arg arg4 = desc->args[4];
#endif

//  ======
//  Timing
//  ======
    double __t1, __t2, __c1, __c2;

    ops_arg args[5];

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;
    args[3] = arg3;
    args[4] = arg4;

#if defined(CHECKPOINTING) && !defined(OPS_LAZY)
    if (!ops_checkpointing_before(args, 5, range, 8)) return;
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timing_realloc(block->instance, 8, "timeloop_eqG");
        block->instance->OPS_kernels[8].count++;
        ops_timers_core(&__c1, &__t1);
    }

#ifdef OPS_DEBUG
    ops_register_args(block->instance, args, "timeloop_eqG");
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
    if (compute_ranges(args, 5, block, range, start, end, arg_idx) < 0) return;
#endif

    int start0 = start[0];
    int end0 = end[0];
    int start1 = start[1];
    int end1 = end[1];

    //  =====================================================
    //  Initialize global variable with the dimension of dats
    //  =====================================================
    int xdim0_timeloop_eqG = args[0].dat->size[0];
    int ydim0_timeloop_eqG = args[0].dat->size[1];
    int xdim1_timeloop_eqG = args[1].dat->size[0];
    int xdim2_timeloop_eqG = args[2].dat->size[0];

//  =======================================================
//  Set up initial pointers and exchange halos if necessary
//  =======================================================
    int base0 = args[0].dat->base_offset;
    double * __restrict__ workArray_p = (double *)(args[0].data_d + base0);

    int base1 = args[1].dat->base_offset;
    double * __restrict__ ux_p = (double *)(args[1].data_d + base1);

    int base2 = args[2].dat->base_offset;
    double * __restrict__ uy_p = (double *)(args[2].data_d + base2);




    int consts_bytes = 0;

    int *arg3h = (int *)args[3].data;
    consts_bytes += ROUND_UP(args[3].dim*sizeof(int));
    int *arg4h = (int *)args[4].data;
    consts_bytes += ROUND_UP(args[4].dim*sizeof(int));

    reallocConstArrays(block->instance,consts_bytes);
    consts_bytes = 0;

    args[3].data = block->instance->OPS_consts_h + consts_bytes;
    args[3].data_d = block->instance->OPS_consts_d + consts_bytes;
    for (int d = 0; d < args[3].dim; d++) {
        ((int *)args[3].data)[d] = arg3h[d];
    }
    consts_bytes += ROUND_UP(args[3].dim*sizeof(int));

    args[4].data = block->instance->OPS_consts_h + consts_bytes;
    args[4].data_d = block->instance->OPS_consts_d + consts_bytes;
    for (int d = 0; d < args[4].dim; d++) {
        ((int *)args[4].data)[d] = arg4h[d];
    }
    consts_bytes += ROUND_UP(args[4].dim*sizeof(int));

    mvConstArraysToDevice(block->instance,consts_bytes);

    int * __restrict__ cx = (int *)args[3].data_d;
    int * __restrict__ cy = (int *)args[4].data_d;

#ifndef OPS_LAZY
//  ==============
//  Halo Exchanges
//  ==============
    ops_H_D_exchanges_device(args, 5);
    ops_halo_exchanges(args, 5,range);
    ops_H_D_exchanges_device(args, 5);
#endif //OPS_LAZY

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[8].mpi_time += __t2 - __t1;
    }

    #pragma omp target teams distribute parallel for collapse(2)
    for (int n_y = start1; n_y < end1; n_y++) {
        for (int n_x = start0; n_x < end0; n_x++) {

#ifdef OPS_SOA
            ACC<double> workArray(9, xdim0_timeloop_eqG, ydim0_timeloop_eqG, workArray_p + n_x*1 + n_y * xdim0_timeloop_eqG*1);
#else
            ACC<double> workArray(9, xdim0_timeloop_eqG, ydim0_timeloop_eqG, workArray_p + 9*(n_x*1 + n_y * xdim0_timeloop_eqG*1));
#endif

const ACC<double> ux(xdim1_timeloop_eqG, ux_p + n_x*1 + n_y * xdim1_timeloop_eqG*1);

const ACC<double> uy(xdim2_timeloop_eqG, uy_p + n_x*1 + n_y * xdim2_timeloop_eqG*1);



    for (int f = 0; f < 9; f++) {
        workArray(f,0,0) = ux(0,0)*cx[f] + uy(0,0)*cy[f];
    }

}
}

if (block->instance->OPS_diags > 1) {
ops_timers_core(&__c1, &__t1);
block->instance->OPS_kernels[8].time += __t1 - __t2;
}
#ifndef OPS_LAZY
    ops_set_dirtybit_device(args, 5);
    ops_set_halo_dirtybit3(&args[0], range);
#endif

    if (block->instance->OPS_diags > 1) {
        //  ====================
        //  Update kernel record
        //  ====================
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[8].mpi_time += __t2 - __t1;
        block->instance->OPS_kernels[8].transfer += ops_compute_transfer(dim, start, end, &arg0);
        block->instance->OPS_kernels[8].transfer += ops_compute_transfer(dim, start, end, &arg1);
        block->instance->OPS_kernels[8].transfer += ops_compute_transfer(dim, start, end, &arg2);
    }
}

#ifdef OPS_LAZY
void ops_par_loop_timeloop_eqG(
    const char *name,
    ops_block block,
    int dim,
    int* range,
    ops_arg arg0,
    ops_arg arg1,
    ops_arg arg2,
    ops_arg arg3,
    ops_arg arg4
)
{
    ops_arg args[5];

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;
    args[3] = arg3;
    args[4] = arg4;

    create_kerneldesc_and_enque("timeloop_eqG", args, 5, 8, dim, 1, range, block, ops_par_loop_timeloop_eqG_execute);
}
#endif
