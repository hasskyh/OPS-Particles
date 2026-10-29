// Auto-generated at 2026-10-29 00:14:11.776470 by ops-translator legacy


//  ==================
//  Host stub function
//  ==================
#ifndef OPS_LAZY
void ops_par_loop_checkop_kernel(
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
void ops_par_loop_checkop_kernel_execute(ops_kernel_descriptor *desc)
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
    if (!ops_checkpointing_before(args,6, range, 14)) return;
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timing_realloc(block->instance, 14, "checkop_kernel");
        block->instance->OPS_kernels[14].count++;
        ops_timers_core(&__c1, &__t1);
    }

#ifdef OPS_DEBUG
    ops_register_args(block->instance, args, "checkop_kernel");
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
    if (compute_ranges(args, 6, block, range, start, end, arg_idx) < 0) return;
#endif


//  =====================================================
//  Initialize global variable with the dimension of dats
//  =====================================================

//  =======================================================
//  Set up initial pointers and exchange halos if necessary
//  =======================================================
    int base0 = args[0].dat->base_offset;
    double * __restrict__ rho_new_p = (double *)(args[0].data_d + base0);
    int base1 = args[1].dat->base_offset;
    double * __restrict__ x_p = (double *)(args[1].data_d + base1);
    int base2 = args[2].dat->base_offset;
    double * __restrict__ rhoin_p = (double *)(args[2].data_d + base2);
#ifdef OPS_MPI
    double * __restrict__ p_a3 = (double *)(((ops_reduction)args[3].data)->data + ((ops_reduction)args[3].data)->size * block->index);
#else //OPS_MPI
    double * __restrict__ p_a3 = (double *)((ops_reduction)args[3].data)->data;
#endif //OPS_MPI

#ifdef OPS_MPI
    double * __restrict__ p_a4 = (double *)(((ops_reduction)args[4].data)->data + ((ops_reduction)args[4].data)->size * block->index);
#else //OPS_MPI
    double * __restrict__ p_a4 = (double *)((ops_reduction)args[4].data)->data;
#endif //OPS_MPI

#ifdef OPS_MPI
    int * __restrict__ p_a5 = (int *)(((ops_reduction)args[5].data)->data + ((ops_reduction)args[5].data)->size * block->index);
#else //OPS_MPI
    int * __restrict__ p_a5 = (int *)((ops_reduction)args[5].data)->data;
#endif //OPS_MPI

    int maxblocks = (end[0]-start[0]-1)/block->instance->OPS_block_size_x+1;
    int reduct_bytes = 0;
    size_t reduct_size = 0;

    reduct_bytes += ROUND_UP(maxblocks*1*sizeof(double));
    reduct_size = MAX(reduct_size,1*sizeof(double));

    reduct_bytes += ROUND_UP(maxblocks*1*sizeof(double));
    reduct_size = MAX(reduct_size,1*sizeof(double));

    reduct_bytes += ROUND_UP(maxblocks*1*sizeof(int));
    reduct_size = MAX(reduct_size,1*sizeof(int));

    reallocReductArrays(block->instance,reduct_bytes);
    reduct_bytes = 0;

    arg3.data = block->instance->OPS_reduct_h + reduct_bytes;
    double *arg3_data_d = (double*)(block->instance->OPS_reduct_d + reduct_bytes);
    for (int b = 0; b < maxblocks; b++) {
        for (int d = 0; d < 1; d++) {
            ((double *)arg3.data)[d+b*1] = ZERO_double;
        }
    }
    reduct_bytes += ROUND_UP(maxblocks*1*sizeof(double));

    arg4.data = block->instance->OPS_reduct_h + reduct_bytes;
    double *arg4_data_d = (double*)(block->instance->OPS_reduct_d + reduct_bytes);
    for (int b = 0; b < maxblocks; b++) {
        for (int d = 0; d < 1; d++) {
            ((double *)arg4.data)[d+b*1] = ZERO_double;
        }
    }
    reduct_bytes += ROUND_UP(maxblocks*1*sizeof(double));

    arg5.data = block->instance->OPS_reduct_h + reduct_bytes;
    int *arg5_data_d = (int*)(block->instance->OPS_reduct_d + reduct_bytes);
    for (int b = 0; b < maxblocks; b++) {
        for (int d = 0; d < 1; d++) {
            ((int *)arg5.data)[d+b*1] = ZERO_int;
        }
    }
    reduct_bytes += ROUND_UP(maxblocks*1*sizeof(int));

    mvReductArraysToDevice(block->instance,reduct_bytes);

#ifndef OPS_LAZY
//  ==============
//  Halo Exchanges
//  ==============
    ops_H_D_exchanges_device(args, 6);
    ops_halo_exchanges(args,6,range);
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[14].mpi_time += __t2 - __t1;
    }

    int start_0 = start[0];
    int end_0 = end[0];
    if ((end[0]-start[0])>0) {
        block->instance->sycl_instance->queue->submit([&](cl::sycl::handler &cgh) {
          cl::sycl::accessor<char, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> local_mem(reduct_size * cl::sycl::range<1>(block->instance->OPS_block_size_x),cgh);

          auto rhol_sycl = (*rhol_p).template get_access<cl::sycl::access::mode::read>(cgh);

          cgh.parallel_for<class checkop_kernel_kernel>(cl::sycl::nd_range<1>(cl::sycl::range<1>(
                ((end[0]-start[0]-1)/block->instance->OPS_block_size_x+1)*block->instance->OPS_block_size_x
                 ),cl::sycl::range<1>(
          block->instance->OPS_block_size_x
                 ))
          , [=](cl::sycl::nd_item<1> item
          ) [[intel::kernel_args_restrict]] {

              int n_x = item.get_global_id()[0]+start_0;

              const ACC<double> rho_new(rho_new_p + n_x*1);
              const ACC<double> x(x_p + n_x*1);
              const ACC<double> rhoin(rhoin_p + n_x*1);
              double pre[1];
              pre[0] = ZERO_double;
              double post[1];
              post[0] = ZERO_double;
              int num[1];
              num[0] = ZERO_int;
//  =========
//  User code
//  =========
                if (n_x < end_0) {
                    
  double diff;
  diff = (rho_new(0) - rhoin(0));
  if(cl::sycl::fabs(diff)<0.01 && x(0) > -4.1){
    *post = *post + diff*diff;
    *num = *num + 1;

  }
  else
    *pre = *pre + (rho_new(0) - rhol_sycl[0])* (rho_new(0) - rhol_sycl[0]);

                }
                int group_size = item.get_local_range(0);
                for (int d = 0; d < 1; d++) {
                    ops_reduction_sycl<OPS_INC>(arg3_data_d + d+item.get_group_linear_id()*1, pre[d], (double*)&local_mem[0], item, group_size);
                }
                for (int d = 0; d < 1; d++) {
                    ops_reduction_sycl<OPS_INC>(arg4_data_d + d+item.get_group_linear_id()*1, post[d], (double*)&local_mem[0], item, group_size);
                }
                for (int d = 0; d < 1; d++) {
                    ops_reduction_sycl<OPS_INC>(arg5_data_d + d+item.get_group_linear_id()*1, num[d], (int*)&local_mem[0], item, group_size);
                }
            });
        });
    }
    mvReductArraysToHost(block->instance,reduct_bytes);
    for (int b = 0; b < maxblocks; b++) {
        for (int d = 0; d < 1; d++) {
            p_a3[d] = p_a3[d] + ((double *)arg3.data)[d+b*1];
        }
    }
    for (int b = 0; b < maxblocks; b++) {
        for (int d = 0; d < 1; d++) {
            p_a4[d] = p_a4[d] + ((double *)arg4.data)[d+b*1];
        }
    }
    for (int b = 0; b < maxblocks; b++) {
        for (int d = 0; d < 1; d++) {
            p_a5[d] = p_a5[d] + ((int *)arg5.data)[d+b*1];
        }
    }

    if (block->instance->OPS_diags > 1) {
        block->instance->sycl_instance->queue->wait();
        ops_timers_core(&__c1, &__t1);
        block->instance->OPS_kernels[14].time += __t1 -__t2;
    }

#ifndef OPS_LAZY
    ops_set_dirtybit_device(args, 6);
#endif

    if (block->instance->OPS_diags > 1) {
//  ====================
//  Update kernel record
//  ====================
    ops_timers_core(&__c2, &__t2);
    block->instance->OPS_kernels[14].mpi_time += __t2 -__t1;
    block->instance->OPS_kernels[14].transfer += ops_compute_transfer(dim, start, end, &arg0);
    block->instance->OPS_kernels[14].transfer += ops_compute_transfer(dim, start, end, &arg1);
    block->instance->OPS_kernels[14].transfer += ops_compute_transfer(dim, start, end, &arg2);
}
}

#ifdef OPS_LAZY
void ops_par_loop_checkop_kernel(
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

    create_kerneldesc_and_enque("checkop_kernel", args, 6, 14, dim, 1, range, block, ops_par_loop_checkop_kernel_execute);
}
#endif
