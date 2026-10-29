// Auto-generated at 2026-10-29 00:14:11.736580 by ops-translator legacy


//  ==================
//  Host stub function
//  ==================
#ifndef OPS_LAZY
void ops_par_loop_limiter_kernel(
    const char *name,
    ops_block block,
    int dim,
    int* range,
    ops_arg arg0, 
    ops_arg arg1, 
    ops_arg arg2
)
{
#else
void ops_par_loop_limiter_kernel_execute(ops_kernel_descriptor *desc)
{
    ops_block block = desc->block;
    int dim = desc->dim;
    int *range = desc->range;
    ops_arg arg0 = desc->args[0];
    ops_arg arg1 = desc->args[1];
    ops_arg arg2 = desc->args[2];
#endif

//  ======
//  Timing
//  ======
    double __t1,__t2,__c1,__c2;

    ops_arg args[3];

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;

#if defined(CHECKPOINTING) && !defined(OPS_LAZY)
    if (!ops_checkpointing_before(args,3, range, 8)) return;
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timing_realloc(block->instance, 8, "limiter_kernel");
        block->instance->OPS_kernels[8].count++;
        ops_timers_core(&__c1, &__t1);
    }

#ifdef OPS_DEBUG
    ops_register_args(block->instance, args, "limiter_kernel");
#endif

//  =================================================
//  compute locally allocated range for the sub-block
//  =================================================
    int start[1];
    int end[1];
#if defined(OPS_MPI) && !defined(OPS_LAZY)
int arg_idx[1];
#endif
#if defined(OPS_LAZY) || !defined(OPS_MPI)
    for (int n = 0; n < 1; n++) {
        start[n] = range[2*n];
        end[n]   = range[2*n+1];
    }
#else
    if (compute_ranges(args, 3, block, range, start, end, arg_idx) < 0) return;
#endif


//  =====================================================
//  Initialize global variable with the dimension of dats
//  =====================================================
    int xdim0_limiter_kernel = args[0].dat->size[0];
    int xdim1_limiter_kernel = args[1].dat->size[0];
    int xdim2_limiter_kernel = args[2].dat->size[0];

//  =======================================================
//  Set up initial pointers and exchange halos if necessary
//  =======================================================
    int base0 = args[0].dat->base_offset;
    double * __restrict__ al_p = (double *)(args[0].data_d + base0);
    int base1 = args[1].dat->base_offset;
    double * __restrict__ tht_p = (double *)(args[1].data_d + base1);
    int base2 = args[2].dat->base_offset;
    double * __restrict__ gt_p = (double *)(args[2].data_d + base2);

#ifndef OPS_LAZY
//  ==============
//  Halo Exchanges
//  ==============
    ops_H_D_exchanges_device(args, 3);
    ops_halo_exchanges(args,3,range);
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[8].mpi_time += __t2 - __t1;
    }

    int start_0 = start[0];
    int end_0 = end[0];
    if ((end[0]-start[0])>0) {
        block->instance->sycl_instance->queue->submit([&](cl::sycl::handler &cgh) {

          auto del2_sycl = (*del2_p).template get_access<cl::sycl::access::mode::read>(cgh);

          cgh.parallel_for<class limiter_kernel_kernel>(cl::sycl::nd_range<1>(cl::sycl::range<1>(
                ((end[0]-start[0]-1)/block->instance->OPS_block_size_x+1)*block->instance->OPS_block_size_x
                 ),cl::sycl::range<1>(
          block->instance->OPS_block_size_x
                 ))
          , [=](cl::sycl::nd_item<1> item
          ) [[intel::kernel_args_restrict]] {

              int n_x = item.get_global_id()[0]+start_0;

              #ifdef OPS_SOA
              const ACC<double> al(3, xdim0_limiter_kernel, al_p + n_x*1);
              #else
              const ACC<double> al(3, xdim0_limiter_kernel, al_p + 3*(n_x*1));
              #endif
              #ifdef OPS_SOA
              ACC<double> tht(3, xdim1_limiter_kernel, tht_p + n_x*1);
              #else
              ACC<double> tht(3, xdim1_limiter_kernel, tht_p + 3*(n_x*1));
              #endif
              #ifdef OPS_SOA
              ACC<double> gt(3, xdim2_limiter_kernel, gt_p + n_x*1);
              #else
              ACC<double> gt(3, xdim2_limiter_kernel, gt_p + 3*(n_x*1));
              #endif
//  =========
//  User code
//  =========
                if (n_x < end_0) {
                    

  double aalm, aal, all, ar, gtt;
  for (int m=0; m < 3 ;m++) {
    aalm = cl::sycl::fabs(al(m,-1));
    aal = cl::sycl::fabs(al(m,0));
    tht(m,0) = cl::sycl::fabs (aal - aalm) / (aal + aalm + del2_sycl[0]);
    all = al(m,-1);
    ar = al(m,0);
    gtt = all * ( ar * ar + del2_sycl[0] ) + ar * (all * all + del2_sycl[0]);
    gt(m,0)= gtt / (ar * ar + all * all + 2.00 * del2_sycl[0]);
  }

                }
            });
        });
    }

    if (block->instance->OPS_diags > 1) {
        block->instance->sycl_instance->queue->wait();
        ops_timers_core(&__c1, &__t1);
        block->instance->OPS_kernels[8].time += __t1 -__t2;
    }

#ifndef OPS_LAZY
    ops_set_dirtybit_device(args, 3);
    ops_set_halo_dirtybit3(&args[1], range);
    ops_set_halo_dirtybit3(&args[2], range);
#endif

    if (block->instance->OPS_diags > 1) {
//  ====================
//  Update kernel record
//  ====================
    ops_timers_core(&__c2, &__t2);
    block->instance->OPS_kernels[8].mpi_time += __t2 -__t1;
    block->instance->OPS_kernels[8].transfer += ops_compute_transfer(dim, start, end, &arg0);
    block->instance->OPS_kernels[8].transfer += ops_compute_transfer(dim, start, end, &arg1);
    block->instance->OPS_kernels[8].transfer += ops_compute_transfer(dim, start, end, &arg2);
}
}

#ifdef OPS_LAZY
void ops_par_loop_limiter_kernel(
    const char *name,
    ops_block block,
    int dim,
    int* range,
    ops_arg arg0,
    ops_arg arg1,
    ops_arg arg2
  )
{
    ops_arg args[3];

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;

    create_kerneldesc_and_enque("limiter_kernel", args, 3, 8, dim, 1, range, block, ops_par_loop_limiter_kernel_execute);
}
#endif
