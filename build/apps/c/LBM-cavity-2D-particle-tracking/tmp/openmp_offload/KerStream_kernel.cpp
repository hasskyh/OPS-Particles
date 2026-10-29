// Auto-generated at 2026-10-29 00:14:13.153891 by ops-translator legacy


//  ==================
//  Host stub function
//  ==================
#ifndef OPS_LAZY
void ops_par_loop_KerStream(
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
void ops_par_loop_KerStream_execute(ops_kernel_descriptor *desc)
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
    if (!ops_checkpointing_before(args, 4, range, 13)) return;
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timing_realloc(block->instance, 13, "KerStream");
        block->instance->OPS_kernels[13].count++;
        ops_timers_core(&__c1, &__t1);
    }

#ifdef OPS_DEBUG
    ops_register_args(block->instance, args, "KerStream");
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
    int xdim0_KerStream = args[0].dat->size[0];
    int ydim0_KerStream = args[0].dat->size[1];
    int xdim1_KerStream = args[1].dat->size[0];
    int ydim1_KerStream = args[1].dat->size[1];

//  =======================================================
//  Set up initial pointers and exchange halos if necessary
//  =======================================================
    int base0 = args[0].dat->base_offset;
    double * __restrict__ f_p = (double *)(args[0].data_d + base0);

    int base1 = args[1].dat->base_offset;
    double * __restrict__ f_work_p = (double *)(args[1].data_d + base1);




    int consts_bytes = 0;

    int *arg2h = (int *)args[2].data;
    consts_bytes += ROUND_UP(args[2].dim*sizeof(int));
    int *arg3h = (int *)args[3].data;
    consts_bytes += ROUND_UP(args[3].dim*sizeof(int));

    reallocConstArrays(block->instance,consts_bytes);
    consts_bytes = 0;

    args[2].data = block->instance->OPS_consts_h + consts_bytes;
    args[2].data_d = block->instance->OPS_consts_d + consts_bytes;
    for (int d = 0; d < args[2].dim; d++) {
        ((int *)args[2].data)[d] = arg2h[d];
    }
    consts_bytes += ROUND_UP(args[2].dim*sizeof(int));

    args[3].data = block->instance->OPS_consts_h + consts_bytes;
    args[3].data_d = block->instance->OPS_consts_d + consts_bytes;
    for (int d = 0; d < args[3].dim; d++) {
        ((int *)args[3].data)[d] = arg3h[d];
    }
    consts_bytes += ROUND_UP(args[3].dim*sizeof(int));

    mvConstArraysToDevice(block->instance,consts_bytes);

    int * __restrict__ cx = (int *)args[2].data_d;
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
        block->instance->OPS_kernels[13].mpi_time += __t2 - __t1;
    }

    #pragma omp target teams distribute parallel for collapse(2)
    for (int n_y = start1; n_y < end1; n_y++) {
        for (int n_x = start0; n_x < end0; n_x++) {

#ifdef OPS_SOA
            ACC<double> f(9, xdim0_KerStream, ydim0_KerStream, f_p + n_x*1 + n_y * xdim0_KerStream*1);
#else
            ACC<double> f(9, xdim0_KerStream, ydim0_KerStream, f_p + 9*(n_x*1 + n_y * xdim0_KerStream*1));
#endif

#ifdef OPS_SOA
const ACC<double> f_work(9, xdim1_KerStream, ydim1_KerStream, f_work_p + n_x*1 + n_y * xdim1_KerStream*1);
#else
const ACC<double> f_work(9, xdim1_KerStream, ydim1_KerStream, f_work_p + 9*(n_x*1 + n_y * xdim1_KerStream*1));
#endif


  for (int i = 0; i < 9; i++)
    f(i, 0, 0) = f_work(i, -cx[i], -cy[i]);

}
}

if (block->instance->OPS_diags > 1) {
ops_timers_core(&__c1, &__t1);
block->instance->OPS_kernels[13].time += __t1 - __t2;
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
        block->instance->OPS_kernels[13].mpi_time += __t2 - __t1;
        block->instance->OPS_kernels[13].transfer += ops_compute_transfer(dim, start, end, &arg0);
        block->instance->OPS_kernels[13].transfer += ops_compute_transfer(dim, start, end, &arg1);
    }
}

#ifdef OPS_LAZY
void ops_par_loop_KerStream(
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

    create_kerneldesc_and_enque("KerStream", args, 4, 13, dim, 1, range, block, ops_par_loop_KerStream_execute);
}
#endif
