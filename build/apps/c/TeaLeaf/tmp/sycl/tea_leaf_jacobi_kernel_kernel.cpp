// Auto-generated at 2026-10-29 00:14:10.808024 by ops-translator legacy


//  ==================
//  Host stub function
//  ==================
#ifndef OPS_LAZY
void ops_par_loop_tea_leaf_jacobi_kernel(
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
void ops_par_loop_tea_leaf_jacobi_kernel_execute(ops_kernel_descriptor *desc)
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
    if (!ops_checkpointing_before(args,8, range, 33)) return;
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timing_realloc(block->instance, 33, "tea_leaf_jacobi_kernel");
        block->instance->OPS_kernels[33].count++;
        ops_timers_core(&__c1, &__t1);
    }

#ifdef OPS_DEBUG
    ops_register_args(block->instance, args, "tea_leaf_jacobi_kernel");
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
    int xdim0_tea_leaf_jacobi_kernel = args[0].dat->size[0];
    int xdim1_tea_leaf_jacobi_kernel = args[1].dat->size[0];
    int xdim2_tea_leaf_jacobi_kernel = args[2].dat->size[0];
    int xdim3_tea_leaf_jacobi_kernel = args[3].dat->size[0];
    int xdim4_tea_leaf_jacobi_kernel = args[4].dat->size[0];

//  =======================================================
//  Set up initial pointers and exchange halos if necessary
//  =======================================================
    int base0 = args[0].dat->base_offset;
    double * __restrict__ u1_p = (double *)(args[0].data_d + base0);
    int base1 = args[1].dat->base_offset;
    double * __restrict__ Kx_p = (double *)(args[1].data_d + base1);
    int base2 = args[2].dat->base_offset;
    double * __restrict__ Ky_p = (double *)(args[2].data_d + base2);
    int base3 = args[3].dat->base_offset;
    double * __restrict__ un_p = (double *)(args[3].data_d + base3);
    int base4 = args[4].dat->base_offset;
    double * __restrict__ u0_p = (double *)(args[4].data_d + base4);
    double rx_val = *(double *)args[5].data;

    double ry_val = *(double *)args[6].data;

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

    reallocReductArrays(block->instance,reduct_bytes);
    reduct_bytes = 0;

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
        block->instance->OPS_kernels[33].mpi_time += __t2 - __t1;
    }

    int start_0 = start[0];
    int end_0 = end[0];
    int start_1 = start[1];
    int end_1 = end[1];
    if ((end[0]-start[0])>0 && (end[1]-start[1])>0) {
        block->instance->sycl_instance->queue->submit([&](cl::sycl::handler &cgh) {
          cl::sycl::accessor<char, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> local_mem(reduct_size * cl::sycl::range<1>(block->instance->OPS_block_size_x*block->instance->OPS_block_size_y),cgh);


          cgh.parallel_for<class tea_leaf_jacobi_kernel_kernel>(cl::sycl::nd_range<2>(cl::sycl::range<2>(
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

              ACC<double> u1(xdim0_tea_leaf_jacobi_kernel, u1_p + n_x*1 + n_y * xdim0_tea_leaf_jacobi_kernel*1);
              const ACC<double> Kx(xdim1_tea_leaf_jacobi_kernel, Kx_p + n_x*1 + n_y * xdim1_tea_leaf_jacobi_kernel*1);
              const ACC<double> Ky(xdim2_tea_leaf_jacobi_kernel, Ky_p + n_x*1 + n_y * xdim2_tea_leaf_jacobi_kernel*1);
              const ACC<double> un(xdim3_tea_leaf_jacobi_kernel, un_p + n_x*1 + n_y * xdim3_tea_leaf_jacobi_kernel*1);
              const ACC<double> u0(xdim4_tea_leaf_jacobi_kernel, u0_p + n_x*1 + n_y * xdim4_tea_leaf_jacobi_kernel*1);
              const double *rx = &rx_val;
              const double *ry = &ry_val;
              double error[1];
              error[0] = ZERO_double;
//  =========
//  User code
//  =========
                if (n_x < end_0 && n_y < end_1) {
                    
	u1(0,0) = (u0(0,0)
		+ (*rx)*(Kx(1, 0) *un(1, 0) + Kx(0,0)*un(-1, 0))
		+ (*ry)*(Ky(0, 1) *un(0, 1) + Ky(0,0)*un(0, -1)))
			/(1.0
				+ (*rx)*(Kx(1, 0) + Kx(0,0))
				+ (*ry)*(Ky(0, 1) + Ky(0,0)));

    *error = *error + cl::sycl::fabs(u1(0,0) - un(0,0));

                }
                int group_size = item.get_local_range(0);
                group_size *= item.get_local_range(1);
                for (int d = 0; d < 1; d++) {
                    ops_reduction_sycl<OPS_INC>(arg7_data_d + d+item.get_group_linear_id()*1, error[d], (double*)&local_mem[0], item, group_size);
                }
            });
        });
    }
    mvReductArraysToHost(block->instance,reduct_bytes);
    for (int b = 0; b < maxblocks; b++) {
        for (int d = 0; d < 1; d++) {
            p_a7[d] = p_a7[d] + ((double *)arg7.data)[d+b*1];
        }
    }

    if (block->instance->OPS_diags > 1) {
        block->instance->sycl_instance->queue->wait();
        ops_timers_core(&__c1, &__t1);
        block->instance->OPS_kernels[33].time += __t1 -__t2;
    }

#ifndef OPS_LAZY
    ops_set_dirtybit_device(args, 8);
    ops_set_halo_dirtybit3(&args[0], range);
#endif

    if (block->instance->OPS_diags > 1) {
//  ====================
//  Update kernel record
//  ====================
    ops_timers_core(&__c2, &__t2);
    block->instance->OPS_kernels[33].mpi_time += __t2 -__t1;
    block->instance->OPS_kernels[33].transfer += ops_compute_transfer(dim, start, end, &arg0);
    block->instance->OPS_kernels[33].transfer += ops_compute_transfer(dim, start, end, &arg1);
    block->instance->OPS_kernels[33].transfer += ops_compute_transfer(dim, start, end, &arg2);
    block->instance->OPS_kernels[33].transfer += ops_compute_transfer(dim, start, end, &arg3);
    block->instance->OPS_kernels[33].transfer += ops_compute_transfer(dim, start, end, &arg4);
}
}

#ifdef OPS_LAZY
void ops_par_loop_tea_leaf_jacobi_kernel(
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

    create_kerneldesc_and_enque("tea_leaf_jacobi_kernel", args, 8, 33, dim, 1, range, block, ops_par_loop_tea_leaf_jacobi_kernel_execute);
}
#endif
