// Auto-generated at 2026-10-29 00:13:52.472700 by ops-translator legacy


//  ==================
//  Host stub function
//  ==================
#ifndef OPS_LAZY
void ops_par_loop_calc_dt_kernel_get(
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
void ops_par_loop_calc_dt_kernel_get_execute(ops_kernel_descriptor *desc)
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
    if (!ops_checkpointing_before(args,4, range, 25)) return;
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timing_realloc(block->instance, 25, "calc_dt_kernel_get");
        block->instance->OPS_kernels[25].count++;
        ops_timers_core(&__c1, &__t1);
    }

#ifdef OPS_DEBUG
    ops_register_args(block->instance, args, "calc_dt_kernel_get");
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
    int xdim0_calc_dt_kernel_get = args[0].dat->size[0];
    int xdim1_calc_dt_kernel_get = args[1].dat->size[0];

//  =======================================================
//  Set up initial pointers and exchange halos if necessary
//  =======================================================
    int base0 = args[0].dat->base_offset;
    double * __restrict__ cellx_p = (double *)(args[0].data_d + base0);
    int base1 = args[1].dat->base_offset;
    double * __restrict__ celly_p = (double *)(args[1].data_d + base1);
#ifdef OPS_MPI
    double * __restrict__ p_a2 = (double *)(((ops_reduction)args[2].data)->data + ((ops_reduction)args[2].data)->size * block->index);
#else //OPS_MPI
    double * __restrict__ p_a2 = (double *)((ops_reduction)args[2].data)->data;
#endif //OPS_MPI

#ifdef OPS_MPI
    double * __restrict__ p_a3 = (double *)(((ops_reduction)args[3].data)->data + ((ops_reduction)args[3].data)->size * block->index);
#else //OPS_MPI
    double * __restrict__ p_a3 = (double *)((ops_reduction)args[3].data)->data;
#endif //OPS_MPI

    int maxblocks = (end[0]-start[0]-1)/block->instance->OPS_block_size_x+1;
    maxblocks *= (end[1]-start[1]-1)/block->instance->OPS_block_size_y+1;
    int reduct_bytes = 0;
    size_t reduct_size = 0;

    reduct_bytes += ROUND_UP(maxblocks*1*sizeof(double));
    reduct_size = MAX(reduct_size,1*sizeof(double));

    reduct_bytes += ROUND_UP(maxblocks*1*sizeof(double));
    reduct_size = MAX(reduct_size,1*sizeof(double));

    reallocReductArrays(block->instance,reduct_bytes);
    reduct_bytes = 0;

    arg2.data = block->instance->OPS_reduct_h + reduct_bytes;
    double *arg2_data_d = (double*)(block->instance->OPS_reduct_d + reduct_bytes);
    for (int b = 0; b < maxblocks; b++) {
        for (int d = 0; d < 1; d++) {
            ((double *)arg2.data)[d+b*1] = ZERO_double;
        }
    }
    reduct_bytes += ROUND_UP(maxblocks*1*sizeof(double));

    arg3.data = block->instance->OPS_reduct_h + reduct_bytes;
    double *arg3_data_d = (double*)(block->instance->OPS_reduct_d + reduct_bytes);
    for (int b = 0; b < maxblocks; b++) {
        for (int d = 0; d < 1; d++) {
            ((double *)arg3.data)[d+b*1] = ZERO_double;
        }
    }
    reduct_bytes += ROUND_UP(maxblocks*1*sizeof(double));

    mvReductArraysToDevice(block->instance,reduct_bytes);

#ifndef OPS_LAZY
//  ==============
//  Halo Exchanges
//  ==============
    ops_H_D_exchanges_device(args, 4);
    ops_halo_exchanges(args,4,range);
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[25].mpi_time += __t2 - __t1;
    }

    int start_0 = start[0];
    int end_0 = end[0];
    int start_1 = start[1];
    int end_1 = end[1];
    if ((end[0]-start[0])>0 && (end[1]-start[1])>0) {
        block->instance->sycl_instance->queue->submit([&](cl::sycl::handler &cgh) {
          cl::sycl::accessor<char, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> local_mem(reduct_size * cl::sycl::range<1>(block->instance->OPS_block_size_x*block->instance->OPS_block_size_y),cgh);


          cgh.parallel_for<class calc_dt_kernel_get_kernel>(cl::sycl::nd_range<2>(cl::sycl::range<2>(
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

              const ACC<double> cellx(xdim0_calc_dt_kernel_get, cellx_p + n_x*1 + n_y * xdim0_calc_dt_kernel_get*0);
              const ACC<double> celly(xdim1_calc_dt_kernel_get, celly_p + n_x*0 + n_y * xdim1_calc_dt_kernel_get*1);
              double xl_pos[1];
              xl_pos[0] = ZERO_double;
              double yl_pos[1];
              yl_pos[0] = ZERO_double;
//  =========
//  User code
//  =========
                if (n_x < end_0 && n_y < end_1) {
                    
  *xl_pos = cellx(0,0);
  *yl_pos = celly(0,0);

                }
                int group_size = item.get_local_range(0);
                group_size *= item.get_local_range(1);
                for (int d = 0; d < 1; d++) {
                    ops_reduction_sycl<OPS_INC>(arg2_data_d + d+item.get_group_linear_id()*1, xl_pos[d], (double*)&local_mem[0], item, group_size);
                }
                for (int d = 0; d < 1; d++) {
                    ops_reduction_sycl<OPS_INC>(arg3_data_d + d+item.get_group_linear_id()*1, yl_pos[d], (double*)&local_mem[0], item, group_size);
                }
            });
        });
    }
    mvReductArraysToHost(block->instance,reduct_bytes);
    for (int b = 0; b < maxblocks; b++) {
        for (int d = 0; d < 1; d++) {
            p_a2[d] = p_a2[d] + ((double *)arg2.data)[d+b*1];
        }
    }
    for (int b = 0; b < maxblocks; b++) {
        for (int d = 0; d < 1; d++) {
            p_a3[d] = p_a3[d] + ((double *)arg3.data)[d+b*1];
        }
    }

    if (block->instance->OPS_diags > 1) {
        block->instance->sycl_instance->queue->wait();
        ops_timers_core(&__c1, &__t1);
        block->instance->OPS_kernels[25].time += __t1 -__t2;
    }

#ifndef OPS_LAZY
    ops_set_dirtybit_device(args, 4);
#endif

    if (block->instance->OPS_diags > 1) {
//  ====================
//  Update kernel record
//  ====================
    ops_timers_core(&__c2, &__t2);
    block->instance->OPS_kernels[25].mpi_time += __t2 -__t1;
    block->instance->OPS_kernels[25].transfer += ops_compute_transfer(dim, start, end, &arg0);
    block->instance->OPS_kernels[25].transfer += ops_compute_transfer(dim, start, end, &arg1);
}
}

#ifdef OPS_LAZY
void ops_par_loop_calc_dt_kernel_get(
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

    create_kerneldesc_and_enque("calc_dt_kernel_get", args, 4, 25, dim, 1, range, block, ops_par_loop_calc_dt_kernel_get_execute);
}
#endif
