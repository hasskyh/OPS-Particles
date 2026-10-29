// Auto-generated at 2026-10-29 00:14:11.716602 by ops-translator legacy


//  ==================
//  Host stub function
//  ==================
#ifndef OPS_LAZY
void ops_par_loop_xder1_kernel(
    const char *name,
    ops_block block,
    int dim,
    int* range,
    ops_arg arg0, 
    ops_arg arg1
)
{
#else
void ops_par_loop_xder1_kernel_execute(ops_kernel_descriptor *desc)
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
    if (!ops_checkpointing_before(args,2, range, 4)) return;
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timing_realloc(block->instance, 4, "xder1_kernel");
        block->instance->OPS_kernels[4].count++;
        ops_timers_core(&__c1, &__t1);
    }

#ifdef OPS_DEBUG
    ops_register_args(block->instance, args, "xder1_kernel");
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
    if (compute_ranges(args, 2, block, range, start, end, arg_idx) < 0) return;
#endif


//  =====================================================
//  Initialize global variable with the dimension of dats
//  =====================================================

//  =======================================================
//  Set up initial pointers and exchange halos if necessary
//  =======================================================
    int base0 = args[0].dat->base_offset;
    double * __restrict__ inp_p = (double *)(args[0].data_d + base0);
    int base1 = args[1].dat->base_offset;
    double * __restrict__ out_p = (double *)(args[1].data_d + base1);

#ifndef OPS_LAZY
//  ==============
//  Halo Exchanges
//  ==============
    ops_H_D_exchanges_device(args, 2);
    ops_halo_exchanges(args,2,range);
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[4].mpi_time += __t2 - __t1;
    }

    int start_0 = start[0];
    int end_0 = end[0];
    if ((end[0]-start[0])>0) {
        block->instance->sycl_instance->queue->submit([&](cl::sycl::handler &cgh) {

          auto dx_sycl = (*dx_p).template get_access<cl::sycl::access::mode::read>(cgh);

          cgh.parallel_for<class xder1_kernel_kernel>(cl::sycl::nd_range<1>(cl::sycl::range<1>(
                ((end[0]-start[0]-1)/block->instance->OPS_block_size_x+1)*block->instance->OPS_block_size_x
                 ),cl::sycl::range<1>(
          block->instance->OPS_block_size_x
                 ))
          , [=](cl::sycl::nd_item<1> item
          ) [[intel::kernel_args_restrict]] {

              int n_x = item.get_global_id()[0]+start_0;

              const ACC<double> inp(inp_p + n_x*1);
              ACC<double> out(out_p + n_x*1);
//  =========
//  User code
//  =========
                if (n_x < end_0) {
                    
  double dix = 1/(12.00*dx_sycl[0]);
  out(0) = (inp(-2) - inp(2)  + 8.0 *(
  inp(1) - inp(-1) )) * dix;

                }
            });
        });
    }

    if (block->instance->OPS_diags > 1) {
        block->instance->sycl_instance->queue->wait();
        ops_timers_core(&__c1, &__t1);
        block->instance->OPS_kernels[4].time += __t1 -__t2;
    }

#ifndef OPS_LAZY
    ops_set_dirtybit_device(args, 2);
    ops_set_halo_dirtybit3(&args[1], range);
#endif

    if (block->instance->OPS_diags > 1) {
//  ====================
//  Update kernel record
//  ====================
    ops_timers_core(&__c2, &__t2);
    block->instance->OPS_kernels[4].mpi_time += __t2 -__t1;
    block->instance->OPS_kernels[4].transfer += ops_compute_transfer(dim, start, end, &arg0);
    block->instance->OPS_kernels[4].transfer += ops_compute_transfer(dim, start, end, &arg1);
}
}

#ifdef OPS_LAZY
void ops_par_loop_xder1_kernel(
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

    create_kerneldesc_and_enque("xder1_kernel", args, 2, 4, dim, 1, range, block, ops_par_loop_xder1_kernel_execute);
}
#endif
