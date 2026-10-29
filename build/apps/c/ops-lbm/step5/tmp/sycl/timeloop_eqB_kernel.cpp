// Auto-generated at 2026-10-29 00:14:12.622256 by ops-translator legacy


//  ==================
//  Host stub function
//  ==================
#ifndef OPS_LAZY
void ops_par_loop_timeloop_eqB(
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
void ops_par_loop_timeloop_eqB_execute(ops_kernel_descriptor *desc)
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
    double __t1,__t2,__c1,__c2;

    ops_arg args[4];

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;
    args[3] = arg3;

#if defined(CHECKPOINTING) && !defined(OPS_LAZY)
    if (!ops_checkpointing_before(args,4, range, 3)) return;
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timing_realloc(block->instance, 3, "timeloop_eqB");
        block->instance->OPS_kernels[3].count++;
        ops_timers_core(&__c1, &__t1);
    }

#ifdef OPS_DEBUG
    ops_register_args(block->instance, args, "timeloop_eqB");
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


//  =====================================================
//  Initialize global variable with the dimension of dats
//  =====================================================
    int xdim0_timeloop_eqB = args[0].dat->size[0];
    int ydim0_timeloop_eqB = args[0].dat->size[1];
    int xdim1_timeloop_eqB = args[1].dat->size[0];
    int ydim1_timeloop_eqB = args[1].dat->size[1];

//  =======================================================
//  Set up initial pointers and exchange halos if necessary
//  =======================================================
    int base0 = args[0].dat->base_offset;
    double * __restrict__ N_p = (double *)(args[0].data_d + base0);
    int base1 = args[1].dat->base_offset;
    double * __restrict__ workArray_p = (double *)(args[1].data_d + base1);
    int *arg2h = (int *)args[2].data;

    int *arg3h = (int *)args[3].data;

    int consts_bytes = 0;

    consts_bytes += ROUND_UP(9*sizeof(int));
    consts_bytes += ROUND_UP(9*sizeof(int));
    reallocConstArrays(block->instance,consts_bytes);
    consts_bytes = 0;
    consts_bytes = 0;
    arg2.data = block->instance->OPS_consts_h + consts_bytes;
    int* arg2_data_d = (int*)( block->instance->OPS_consts_d + consts_bytes);
    for (int d=0; d<9; d++) ((int *)arg2.data)[d] = arg2h[d];
    consts_bytes += ROUND_UP(9*sizeof(int));
    consts_bytes = 0;
    arg3.data = block->instance->OPS_consts_h + consts_bytes;
    int* arg3_data_d = (int*)( block->instance->OPS_consts_d + consts_bytes);
    for (int d=0; d<9; d++) ((int *)arg3.data)[d] = arg3h[d];
    consts_bytes += ROUND_UP(9*sizeof(int));
    mvConstArraysToDevice(block->instance,consts_bytes);

#ifndef OPS_LAZY
//  ==============
//  Halo Exchanges
//  ==============
    ops_H_D_exchanges_device(args, 4);
    ops_halo_exchanges(args,4,range);
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[3].mpi_time += __t2 - __t1;
    }

    int start_0 = start[0];
    int end_0 = end[0];
    int start_1 = start[1];
    int end_1 = end[1];
    if ((end[0]-start[0])>0 && (end[1]-start[1])>0) {
        block->instance->sycl_instance->queue->submit([&](cl::sycl::handler &cgh) {


          cgh.parallel_for<class timeloop_eqB_kernel>(cl::sycl::nd_range<2>(cl::sycl::range<2>(
               ((end[1]-start[1]-1)/block->instance->OPS_block_size_y+1)*block->instance->OPS_block_size_y,
                ((end[0]-start[0]-1)/block->instance->OPS_block_size_x+1)*block->instance->OPS_block_size_x
                 ),cl::sycl::range<2>(
                 block->instance->OPS_block_size_y,
          block->instance->OPS_block_size_x
                 ))
          , [=](cl::sycl::nd_item<2> item
          ) [[intel::kernel_args_restrict]] {

              int n_y = item.get_global_id()[0]+start_1;
              int n_x = item.get_global_id()[1]+start_0;

              #ifdef OPS_SOA
              ACC<double> N(9, xdim0_timeloop_eqB, ydim0_timeloop_eqB, N_p + n_x*1 + n_y * xdim0_timeloop_eqB*1);
              #else
              ACC<double> N(9, xdim0_timeloop_eqB, ydim0_timeloop_eqB, N_p + 9*(n_x*1 + n_y * xdim0_timeloop_eqB*1));
              #endif
              #ifdef OPS_SOA
              const ACC<double> workArray(9, xdim1_timeloop_eqB, ydim1_timeloop_eqB, workArray_p + n_x*1 + n_y * xdim1_timeloop_eqB*1);
              #else
              const ACC<double> workArray(9, xdim1_timeloop_eqB, ydim1_timeloop_eqB, workArray_p + 9*(n_x*1 + n_y * xdim1_timeloop_eqB*1));
              #endif
              const int *cx = arg2_data_d;
              const int *cy = arg3_data_d;
//  =========
//  User code
//  =========
                if (n_x < end_0 && n_y < end_1) {
                    

    for (int f = 1; f < 9; f++) {
        N(f,0,0) = workArray(f,-cx[f],-cy[f]);
    }

                }
            });
        });
    }

    if (block->instance->OPS_diags > 1) {
        block->instance->sycl_instance->queue->wait();
        ops_timers_core(&__c1, &__t1);
        block->instance->OPS_kernels[3].time += __t1 -__t2;
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
    block->instance->OPS_kernels[3].mpi_time += __t2 -__t1;
    block->instance->OPS_kernels[3].transfer += ops_compute_transfer(dim, start, end, &arg0);
    block->instance->OPS_kernels[3].transfer += ops_compute_transfer(dim, start, end, &arg1);
}
}

#ifdef OPS_LAZY
void ops_par_loop_timeloop_eqB(
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

    create_kerneldesc_and_enque("timeloop_eqB", args, 4, 3, dim, 1, range, block, ops_par_loop_timeloop_eqB_execute);
}
#endif
