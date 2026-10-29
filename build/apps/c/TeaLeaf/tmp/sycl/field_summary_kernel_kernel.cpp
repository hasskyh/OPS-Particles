// Auto-generated at 2026-10-29 00:14:10.587785 by ops-translator legacy


//  ==================
//  Host stub function
//  ==================
#ifndef OPS_LAZY
void ops_par_loop_field_summary_kernel(
    const char *name,
    ops_block block,
    int dim,
    int* range,
    ops_arg arg0, 
    ops_arg arg1, 
    ops_arg arg2, 
    ops_arg arg3, 
    ops_arg arg4, 
    ops_arg arg5, 
    ops_arg arg6, 
    ops_arg arg7
)
{
#else
void ops_par_loop_field_summary_kernel_execute(ops_kernel_descriptor *desc)
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
    ops_arg arg6 = desc->args[6];
    ops_arg arg7 = desc->args[7];
#endif

//  ======
//  Timing
//  ======
    double __t1,__t2,__c1,__c2;

    ops_arg args[8];

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;
    args[3] = arg3;
    args[4] = arg4;
    args[5] = arg5;
    args[6] = arg6;
    args[7] = arg7;

#if defined(CHECKPOINTING) && !defined(OPS_LAZY)
    if (!ops_checkpointing_before(args,8, range, 0)) return;
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timing_realloc(block->instance, 0, "field_summary_kernel");
        block->instance->OPS_kernels[0].count++;
        ops_timers_core(&__c1, &__t1);
    }

#ifdef OPS_DEBUG
    ops_register_args(block->instance, args, "field_summary_kernel");
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
    if (compute_ranges(args, 8, block, range, start, end, arg_idx) < 0) return;
#endif


//  =====================================================
//  Initialize global variable with the dimension of dats
//  =====================================================
    int xdim0_field_summary_kernel = args[0].dat->size[0];
    int xdim1_field_summary_kernel = args[1].dat->size[0];
    int xdim2_field_summary_kernel = args[2].dat->size[0];
    int xdim3_field_summary_kernel = args[3].dat->size[0];

//  =======================================================
//  Set up initial pointers and exchange halos if necessary
//  =======================================================
    int base0 = args[0].dat->base_offset;
    double * __restrict__ volume_p = (double *)(args[0].data_d + base0);
    int base1 = args[1].dat->base_offset;
    double * __restrict__ density_p = (double *)(args[1].data_d + base1);
    int base2 = args[2].dat->base_offset;
    double * __restrict__ energy_p = (double *)(args[2].data_d + base2);
    int base3 = args[3].dat->base_offset;
    double * __restrict__ u_p = (double *)(args[3].data_d + base3);
#ifdef OPS_MPI
    double * __restrict__ p_a4 = (double *)(((ops_reduction)args[4].data)->data + ((ops_reduction)args[4].data)->size * block->index);
#else //OPS_MPI
    double * __restrict__ p_a4 = (double *)((ops_reduction)args[4].data)->data;
#endif //OPS_MPI

#ifdef OPS_MPI
    double * __restrict__ p_a5 = (double *)(((ops_reduction)args[5].data)->data + ((ops_reduction)args[5].data)->size * block->index);
#else //OPS_MPI
    double * __restrict__ p_a5 = (double *)((ops_reduction)args[5].data)->data;
#endif //OPS_MPI

#ifdef OPS_MPI
    double * __restrict__ p_a6 = (double *)(((ops_reduction)args[6].data)->data + ((ops_reduction)args[6].data)->size * block->index);
#else //OPS_MPI
    double * __restrict__ p_a6 = (double *)((ops_reduction)args[6].data)->data;
#endif //OPS_MPI

#ifdef OPS_MPI
    double * __restrict__ p_a7 = (double *)(((ops_reduction)args[7].data)->data + ((ops_reduction)args[7].data)->size * block->index);
#else //OPS_MPI
    double * __restrict__ p_a7 = (double *)((ops_reduction)args[7].data)->data;
#endif //OPS_MPI

    int maxblocks = (end[0]-start[0]-1)/block->instance->OPS_block_size_x+1;
    maxblocks *= (end[1]-start[1]-1)/block->instance->OPS_block_size_y+1;
    int reduct_bytes = 0;
    size_t reduct_size = 0;

    reduct_bytes += ROUND_UP(maxblocks*1*sizeof(double));
    reduct_size = MAX(reduct_size,1*sizeof(double));

    reduct_bytes += ROUND_UP(maxblocks*1*sizeof(double));
    reduct_size = MAX(reduct_size,1*sizeof(double));

    reduct_bytes += ROUND_UP(maxblocks*1*sizeof(double));
    reduct_size = MAX(reduct_size,1*sizeof(double));

    reduct_bytes += ROUND_UP(maxblocks*1*sizeof(double));
    reduct_size = MAX(reduct_size,1*sizeof(double));

    reallocReductArrays(block->instance,reduct_bytes);
    reduct_bytes = 0;

    arg4.data = block->instance->OPS_reduct_h + reduct_bytes;
    double *arg4_data_d = (double*)(block->instance->OPS_reduct_d + reduct_bytes);
    for (int b = 0; b < maxblocks; b++) {
        for (int d = 0; d < 1; d++) {
            ((double *)arg4.data)[d+b*1] = ZERO_double;
        }
    }
    reduct_bytes += ROUND_UP(maxblocks*1*sizeof(double));

    arg5.data = block->instance->OPS_reduct_h + reduct_bytes;
    double *arg5_data_d = (double*)(block->instance->OPS_reduct_d + reduct_bytes);
    for (int b = 0; b < maxblocks; b++) {
        for (int d = 0; d < 1; d++) {
            ((double *)arg5.data)[d+b*1] = ZERO_double;
        }
    }
    reduct_bytes += ROUND_UP(maxblocks*1*sizeof(double));

    arg6.data = block->instance->OPS_reduct_h + reduct_bytes;
    double *arg6_data_d = (double*)(block->instance->OPS_reduct_d + reduct_bytes);
    for (int b = 0; b < maxblocks; b++) {
        for (int d = 0; d < 1; d++) {
            ((double *)arg6.data)[d+b*1] = ZERO_double;
        }
    }
    reduct_bytes += ROUND_UP(maxblocks*1*sizeof(double));

    arg7.data = block->instance->OPS_reduct_h + reduct_bytes;
    double *arg7_data_d = (double*)(block->instance->OPS_reduct_d + reduct_bytes);
    for (int b = 0; b < maxblocks; b++) {
        for (int d = 0; d < 1; d++) {
            ((double *)arg7.data)[d+b*1] = ZERO_double;
        }
    }
    reduct_bytes += ROUND_UP(maxblocks*1*sizeof(double));

    mvReductArraysToDevice(block->instance,reduct_bytes);

#ifndef OPS_LAZY
//  ==============
//  Halo Exchanges
//  ==============
    ops_H_D_exchanges_device(args, 8);
    ops_halo_exchanges(args,8,range);
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[0].mpi_time += __t2 - __t1;
    }

    int start_0 = start[0];
    int end_0 = end[0];
    int start_1 = start[1];
    int end_1 = end[1];
    if ((end[0]-start[0])>0 && (end[1]-start[1])>0) {
        block->instance->sycl_instance->queue->submit([&](cl::sycl::handler &cgh) {
          cl::sycl::accessor<char, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> local_mem(reduct_size * cl::sycl::range<1>(block->instance->OPS_block_size_x*block->instance->OPS_block_size_y),cgh);


          cgh.parallel_for<class field_summary_kernel_kernel>(cl::sycl::nd_range<2>(cl::sycl::range<2>(
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

              const ACC<double> volume(xdim0_field_summary_kernel, volume_p + n_x*1 + n_y * xdim0_field_summary_kernel*1);
              const ACC<double> density(xdim1_field_summary_kernel, density_p + n_x*1 + n_y * xdim1_field_summary_kernel*1);
              const ACC<double> energy(xdim2_field_summary_kernel, energy_p + n_x*1 + n_y * xdim2_field_summary_kernel*1);
              const ACC<double> u(xdim3_field_summary_kernel, u_p + n_x*1 + n_y * xdim3_field_summary_kernel*1);
              double vol[1];
              vol[0] = ZERO_double;
              double mass[1];
              mass[0] = ZERO_double;
              double ie[1];
              ie[0] = ZERO_double;
              double temp[1];
              temp[0] = ZERO_double;
//  =========
//  User code
//  =========
                if (n_x < end_0 && n_y < end_1) {
                    

  double cell_vol, cell_mass;


  cell_vol = volume(0,0);
  cell_mass = cell_vol * density(0,0);
  *vol = *vol + cell_vol;
  *mass = *mass + cell_mass;
  *ie = *ie + cell_mass * energy(0,0);
  *temp = *temp + cell_mass * u(0,0);

                }
                int group_size = item.get_local_range(0);
                group_size *= item.get_local_range(1);
                for (int d = 0; d < 1; d++) {
                    ops_reduction_sycl<OPS_INC>(arg4_data_d + d+item.get_group_linear_id()*1, vol[d], (double*)&local_mem[0], item, group_size);
                }
                for (int d = 0; d < 1; d++) {
                    ops_reduction_sycl<OPS_INC>(arg5_data_d + d+item.get_group_linear_id()*1, mass[d], (double*)&local_mem[0], item, group_size);
                }
                for (int d = 0; d < 1; d++) {
                    ops_reduction_sycl<OPS_INC>(arg6_data_d + d+item.get_group_linear_id()*1, ie[d], (double*)&local_mem[0], item, group_size);
                }
                for (int d = 0; d < 1; d++) {
                    ops_reduction_sycl<OPS_INC>(arg7_data_d + d+item.get_group_linear_id()*1, temp[d], (double*)&local_mem[0], item, group_size);
                }
            });
        });
    }
    mvReductArraysToHost(block->instance,reduct_bytes);
    for (int b = 0; b < maxblocks; b++) {
        for (int d = 0; d < 1; d++) {
            p_a4[d] = p_a4[d] + ((double *)arg4.data)[d+b*1];
        }
    }
    for (int b = 0; b < maxblocks; b++) {
        for (int d = 0; d < 1; d++) {
            p_a5[d] = p_a5[d] + ((double *)arg5.data)[d+b*1];
        }
    }
    for (int b = 0; b < maxblocks; b++) {
        for (int d = 0; d < 1; d++) {
            p_a6[d] = p_a6[d] + ((double *)arg6.data)[d+b*1];
        }
    }
    for (int b = 0; b < maxblocks; b++) {
        for (int d = 0; d < 1; d++) {
            p_a7[d] = p_a7[d] + ((double *)arg7.data)[d+b*1];
        }
    }

    if (block->instance->OPS_diags > 1) {
        block->instance->sycl_instance->queue->wait();
        ops_timers_core(&__c1, &__t1);
        block->instance->OPS_kernels[0].time += __t1 -__t2;
    }

#ifndef OPS_LAZY
    ops_set_dirtybit_device(args, 8);
#endif

    if (block->instance->OPS_diags > 1) {
//  ====================
//  Update kernel record
//  ====================
    ops_timers_core(&__c2, &__t2);
    block->instance->OPS_kernels[0].mpi_time += __t2 -__t1;
    block->instance->OPS_kernels[0].transfer += ops_compute_transfer(dim, start, end, &arg0);
    block->instance->OPS_kernels[0].transfer += ops_compute_transfer(dim, start, end, &arg1);
    block->instance->OPS_kernels[0].transfer += ops_compute_transfer(dim, start, end, &arg2);
    block->instance->OPS_kernels[0].transfer += ops_compute_transfer(dim, start, end, &arg3);
}
}

#ifdef OPS_LAZY
void ops_par_loop_field_summary_kernel(
    const char *name,
    ops_block block,
    int dim,
    int* range,
    ops_arg arg0,
    ops_arg arg1,
    ops_arg arg2,
    ops_arg arg3,
    ops_arg arg4,
    ops_arg arg5,
    ops_arg arg6,
    ops_arg arg7
  )
{
    ops_arg args[8];

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;
    args[3] = arg3;
    args[4] = arg4;
    args[5] = arg5;
    args[6] = arg6;
    args[7] = arg7;

    create_kerneldesc_and_enque("field_summary_kernel", args, 8, 0, dim, 1, range, block, ops_par_loop_field_summary_kernel_execute);
}
#endif
