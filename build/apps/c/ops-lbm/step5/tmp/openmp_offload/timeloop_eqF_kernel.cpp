// Auto-generated at 2026-10-29 00:14:12.302494 by ops-translator legacy


//  ==================
//  Host stub function
//  ==================
#ifndef OPS_LAZY
void ops_par_loop_timeloop_eqF(
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
void ops_par_loop_timeloop_eqF_execute(ops_kernel_descriptor *desc)
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
    if (!ops_checkpointing_before(args, 4, range, 7)) return;
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timing_realloc(block->instance, 7, "timeloop_eqF");
        block->instance->OPS_kernels[7].count++;
        ops_timers_core(&__c1, &__t1);
    }

#ifdef OPS_DEBUG
    ops_register_args(block->instance, args, "timeloop_eqF");
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
    if (compute_ranges(args, 4, block, range, start, end, arg_idx) < 0) return;
#endif

    int start0 = start[0];
    int end0 = end[0];
    int start1 = start[1];
    int end1 = end[1];

    //  =====================================================
    //  Initialize global variable with the dimension of dats
    //  =====================================================
    int xdim0_timeloop_eqF = args[0].dat->size[0];
    int xdim1_timeloop_eqF = args[1].dat->size[0];
    int ydim1_timeloop_eqF = args[1].dat->size[1];
    int xdim2_timeloop_eqF = args[2].dat->size[0];

//  =======================================================
//  Set up initial pointers and exchange halos if necessary
//  =======================================================
    int base0 = args[0].dat->base_offset;
    double * __restrict__ uy_p = (double *)(args[0].data_d + base0);

    int base1 = args[1].dat->base_offset;
    double * __restrict__ N_p = (double *)(args[1].data_d + base1);

    int base2 = args[2].dat->base_offset;
    double * __restrict__ rho_p = (double *)(args[2].data_d + base2);



    int consts_bytes = 0;

    int *arg3h = (int *)args[3].data;
    consts_bytes += ROUND_UP(args[3].dim*sizeof(int));

    reallocConstArrays(block->instance,consts_bytes);
    consts_bytes = 0;

    args[3].data = block->instance->OPS_consts_h + consts_bytes;
    args[3].data_d = block->instance->OPS_consts_d + consts_bytes;
    for (int d = 0; d < args[3].dim; d++) {
        ((int *)args[3].data)[d] = arg3h[d];
    }
    consts_bytes += ROUND_UP(args[3].dim*sizeof(int));

    mvConstArraysToDevice(block->instance,consts_bytes);

    int * __restrict__ cy = (int *)args[3].data_d;

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
        block->instance->OPS_kernels[7].mpi_time += __t2 - __t1;
    }

    #pragma omp target teams distribute parallel for collapse(2)
    for (int n_y = start1; n_y < end1; n_y++) {
        for (int n_x = start0; n_x < end0; n_x++) {

            ACC<double> uy(xdim0_timeloop_eqF, uy_p + n_x*1 + n_y * xdim0_timeloop_eqF*1);

#ifdef OPS_SOA
            const ACC<double> N(9, xdim1_timeloop_eqF, ydim1_timeloop_eqF, N_p + n_x*1 + n_y * xdim1_timeloop_eqF*1);
#else
            const ACC<double> N(9, xdim1_timeloop_eqF, ydim1_timeloop_eqF, N_p + 9*(n_x*1 + n_y * xdim1_timeloop_eqF*1));
#endif

const ACC<double> rho(xdim2_timeloop_eqF, rho_p + n_x*1 + n_y * xdim2_timeloop_eqF*1);



    uy(0,0) = 0.0;
    for (int f = 0; f < 9; f++) {
        uy(0,0) += N(f,0,0) * cy[f];
    }
    uy(0,0) = uy(0,0) / rho(0,0);

}
}

if (block->instance->OPS_diags > 1) {
ops_timers_core(&__c1, &__t1);
block->instance->OPS_kernels[7].time += __t1 - __t2;
}
#ifndef OPS_LAZY
    ops_set_dirtybit_device(args, 4);
    ops_set_halo_dirtybit3(&args[0], range);
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
void ops_par_loop_timeloop_eqF(
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

    create_kerneldesc_and_enque("timeloop_eqF", args, 4, 7, dim, 1, range, block, ops_par_loop_timeloop_eqF_execute);
}
#endif
