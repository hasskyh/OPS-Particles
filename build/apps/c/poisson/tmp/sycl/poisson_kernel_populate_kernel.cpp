// Auto-generated at 2026-10-29 00:13:46.553740 by ops-translator legacy


//  ==================
//  Host stub function
//  ==================
#ifndef OPS_LAZY
void ops_par_loop_poisson_kernel_populate(
    const char *name,
    ops_block block,
    int dim,
    int* range,
    ops_arg arg0, 
    ops_arg arg1, 
    ops_arg arg2, 
    ops_arg arg3, 
    ops_arg arg4, 
    ops_arg arg5
)
{
#else
void ops_par_loop_poisson_kernel_populate_execute(ops_kernel_descriptor *desc)
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
#endif

//  ======
//  Timing
//  ======
    double __t1,__t2,__c1,__c2;

    ops_arg args[6];

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;
    args[3] = arg3;
    args[4] = arg4;
    args[5] = arg5;

#if defined(CHECKPOINTING) && !defined(OPS_LAZY)
    if (!ops_checkpointing_before(args,6, range, 0)) return;
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timing_realloc(block->instance, 0, "poisson_kernel_populate");
        block->instance->OPS_kernels[0].count++;
        ops_timers_core(&__c1, &__t1);
    }

#ifdef OPS_DEBUG
    ops_register_args(block->instance, args, "poisson_kernel_populate");
#endif

//  =================================================
//  compute locally allocated range for the sub-block
//  =================================================
    int start[2];
    int end[2];
    int arg_idx[2];
#if defined(OPS_LAZY) || !defined(OPS_MPI)
    for (int n = 0; n < 2; n++) {
        start[n] = range[2*n];
        end[n]   = range[2*n+1];
    }
#else
    if (compute_ranges(args, 6, block, range, start, end, arg_idx) < 0) return;
#endif

#if defined(OPS_MPI)
#if defined(OPS_LAZY)
    sub_block_list sb = OPS_sub_block_list[block->index];
    arg_idx[0] = sb->decomp_disp[0];
    arg_idx[1] = sb->decomp_disp[1];
#else
    arg_idx[0] -= start[0];
    arg_idx[1] -= start[1];
#endif
#else //OPS_MPI
    arg_idx[0] = 0;
    arg_idx[1] = 0;
#endif //OPS_MPI

//  =====================================================
//  Initialize global variable with the dimension of dats
//  =====================================================
    int xdim3_poisson_kernel_populate = args[3].dat->size[0];
    int xdim4_poisson_kernel_populate = args[4].dat->size[0];
    int xdim5_poisson_kernel_populate = args[5].dat->size[0];

//  =======================================================
//  Set up initial pointers and exchange halos if necessary
//  =======================================================
    int dispx_val = *(int *)args[0].data;

    int dispy_val = *(int *)args[1].data;

    int base3 = args[3].dat->base_offset;
    double * __restrict__ u_p = (double *)(args[3].data_d + base3);
    int base4 = args[4].dat->base_offset;
    double * __restrict__ f_p = (double *)(args[4].data_d + base4);
    int base5 = args[5].dat->base_offset;
    double * __restrict__ ref_p = (double *)(args[5].data_d + base5);

#ifndef OPS_LAZY
//  ==============
//  Halo Exchanges
//  ==============
    ops_H_D_exchanges_device(args, 6);
    ops_halo_exchanges(args,6,range);
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[0].mpi_time += __t2 - __t1;
    }

    int start_0 = start[0];
    int end_0 = end[0];
    int arg_idx_0 = arg_idx[0];
    int start_1 = start[1];
    int end_1 = end[1];
    int arg_idx_1 = arg_idx[1];
    if ((end[0]-start[0])>0 && (end[1]-start[1])>0) {
        block->instance->sycl_instance->queue->submit([&](cl::sycl::handler &cgh) {

          auto dx_sycl = (*dx_p).template get_access<cl::sycl::access::mode::read>(cgh);
          auto dy_sycl = (*dy_p).template get_access<cl::sycl::access::mode::read>(cgh);

          cgh.parallel_for<class poisson_kernel_populate_kernel>(cl::sycl::nd_range<2>(cl::sycl::range<2>(
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

              int idx[] = {arg_idx_0+n_x, arg_idx_1+n_y};

              ACC<double> u(xdim3_poisson_kernel_populate, u_p + n_x*1 + n_y * xdim3_poisson_kernel_populate*1);
              ACC<double> f(xdim4_poisson_kernel_populate, f_p + n_x*1 + n_y * xdim4_poisson_kernel_populate*1);
              ACC<double> ref(xdim5_poisson_kernel_populate, ref_p + n_x*1 + n_y * xdim5_poisson_kernel_populate*1);
              const int *dispx = &dispx_val;
              const int *dispy = &dispy_val;
//  =========
//  User code
//  =========
                if (n_x < end_0 && n_y < end_1) {
                    
  double x = dx_sycl[0] * (double)(idx[0]+dispx[0]);
  double y = dy_sycl[0] * (double)(idx[1]+dispy[0]);

  u(0,0) = myfun(cl::sycl::sin(M_PI*x),cl::sycl::cos(2.0*M_PI*y))-1.0;
  f(0,0) = -5.0*M_PI*M_PI*cl::sycl::sin(M_PI*x)*cl::sycl::cos(2.0*M_PI*y);
  ref(0,0) = cl::sycl::sin(M_PI*x)*cl::sycl::cos(2.0*M_PI*y);


                }
            });
        });
    }

    if (block->instance->OPS_diags > 1) {
        block->instance->sycl_instance->queue->wait();
        ops_timers_core(&__c1, &__t1);
        block->instance->OPS_kernels[0].time += __t1 -__t2;
    }

#ifndef OPS_LAZY
    ops_set_dirtybit_device(args, 6);
    ops_set_halo_dirtybit3(&args[3], range);
    ops_set_halo_dirtybit3(&args[4], range);
    ops_set_halo_dirtybit3(&args[5], range);
#endif

    if (block->instance->OPS_diags > 1) {
//  ====================
//  Update kernel record
//  ====================
    ops_timers_core(&__c2, &__t2);
    block->instance->OPS_kernels[0].mpi_time += __t2 -__t1;
    block->instance->OPS_kernels[0].transfer += ops_compute_transfer(dim, start, end, &arg3);
    block->instance->OPS_kernels[0].transfer += ops_compute_transfer(dim, start, end, &arg4);
    block->instance->OPS_kernels[0].transfer += ops_compute_transfer(dim, start, end, &arg5);
}
}

#ifdef OPS_LAZY
void ops_par_loop_poisson_kernel_populate(
    const char *name,
    ops_block block,
    int dim,
    int* range,
    ops_arg arg0,
    ops_arg arg1,
    ops_arg arg2,
    ops_arg arg3,
    ops_arg arg4,
    ops_arg arg5
  )
{
    ops_arg args[6];

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;
    args[3] = arg3;
    args[4] = arg4;
    args[5] = arg5;

    create_kerneldesc_and_enque("poisson_kernel_populate", args, 6, 0, dim, 1, range, block, ops_par_loop_poisson_kernel_populate_execute);
}
#endif
