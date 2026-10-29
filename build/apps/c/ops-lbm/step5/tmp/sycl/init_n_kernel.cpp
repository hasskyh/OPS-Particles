// Auto-generated at 2026-10-29 00:14:12.612833 by ops-translator legacy


//  ==================
//  Host stub function
//  ==================
#ifndef OPS_LAZY
void ops_par_loop_init_n(
    const char *name,
    ops_block block,
    int dim,
    int* range,
    ops_arg arg0, 
    ops_arg arg1
)
{
#else
void ops_par_loop_init_n_execute(ops_kernel_descriptor *desc)
{
    ops_block block = desc->block;
    int dim = desc->dim;
    int *range = desc->range;
    ops_arg arg0 = desc->args[0];
    ops_arg arg1 = desc->args[1];
#endif

//  ======
//  Timing
//  ======
    double __t1,__t2,__c1,__c2;

    ops_arg args[2];

    args[0] = arg0;
    args[1] = arg1;

#if defined(CHECKPOINTING) && !defined(OPS_LAZY)
    if (!ops_checkpointing_before(args,2, range, 1)) return;
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timing_realloc(block->instance, 1, "init_n");
        block->instance->OPS_kernels[1].count++;
        ops_timers_core(&__c1, &__t1);
    }

#ifdef OPS_DEBUG
    ops_register_args(block->instance, args, "init_n");
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
    if (compute_ranges(args, 2, block, range, start, end, arg_idx) < 0) return;
#endif


//  =====================================================
//  Initialize global variable with the dimension of dats
//  =====================================================
    int xdim0_init_n = args[0].dat->size[0];
    int ydim0_init_n = args[0].dat->size[1];

//  =======================================================
//  Set up initial pointers and exchange halos if necessary
//  =======================================================
    int base0 = args[0].dat->base_offset;
    double * __restrict__ N_p = (double *)(args[0].data_d + base0);
    double *arg1h = (double *)args[1].data;

    int consts_bytes = 0;

    consts_bytes += ROUND_UP(9*sizeof(double));
    reallocConstArrays(block->instance,consts_bytes);
    consts_bytes = 0;
    consts_bytes = 0;
    arg1.data = block->instance->OPS_consts_h + consts_bytes;
    double* arg1_data_d = (double*)( block->instance->OPS_consts_d + consts_bytes);
    for (int d=0; d<9; d++) ((double *)arg1.data)[d] = arg1h[d];
    consts_bytes += ROUND_UP(9*sizeof(double));
    mvConstArraysToDevice(block->instance,consts_bytes);

#ifndef OPS_LAZY
//  ==============
//  Halo Exchanges
//  ==============
    ops_H_D_exchanges_device(args, 2);
    ops_halo_exchanges(args,2,range);
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[1].mpi_time += __t2 - __t1;
    }

    int start_0 = start[0];
    int end_0 = end[0];
    int start_1 = start[1];
    int end_1 = end[1];
    if ((end[0]-start[0])>0 && (end[1]-start[1])>0) {
        block->instance->sycl_instance->queue->submit([&](cl::sycl::handler &cgh) {

          auto rho0_sycl = (*rho0_p).template get_access<cl::sycl::access::mode::read>(cgh);

          cgh.parallel_for<class init_n_kernel>(cl::sycl::nd_range<2>(cl::sycl::range<2>(
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
              ACC<double> N(9, xdim0_init_n, ydim0_init_n, N_p + n_x*1 + n_y * xdim0_init_n*1);
              #else
              ACC<double> N(9, xdim0_init_n, ydim0_init_n, N_p + 9*(n_x*1 + n_y * xdim0_init_n*1));
              #endif
              const double *W = arg1_data_d;
//  =========
//  User code
//  =========
                if (n_x < end_0 && n_y < end_1) {
                    

    for (int f = 0; f < 9; f++) {
        N(f,0,0) = rho0_sycl[0] * W[f];
    }

                }
            });
        });
    }

    if (block->instance->OPS_diags > 1) {
        block->instance->sycl_instance->queue->wait();
        ops_timers_core(&__c1, &__t1);
        block->instance->OPS_kernels[1].time += __t1 -__t2;
    }

#ifndef OPS_LAZY
    ops_set_dirtybit_device(args, 2);
    ops_set_halo_dirtybit3(&args[0], range);
#endif

    if (block->instance->OPS_diags > 1) {
//  ====================
//  Update kernel record
//  ====================
    ops_timers_core(&__c2, &__t2);
    block->instance->OPS_kernels[1].mpi_time += __t2 -__t1;
    block->instance->OPS_kernels[1].transfer += ops_compute_transfer(dim, start, end, &arg0);
}
}

#ifdef OPS_LAZY
void ops_par_loop_init_n(
    const char *name,
    ops_block block,
    int dim,
    int* range,
    ops_arg arg0,
    ops_arg arg1
  )
{
    ops_arg args[2];

    args[0] = arg0;
    args[1] = arg1;

    create_kerneldesc_and_enque("init_n", args, 2, 1, dim, 1, range, block, ops_par_loop_init_n_execute);
}
#endif
