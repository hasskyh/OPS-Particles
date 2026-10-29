// Auto-generated at 2026-10-29 00:14:10.759831 by ops-translator legacy


//  ==================
//  Host stub function
//  ==================
#ifndef OPS_LAZY
void ops_par_loop_tea_leaf_common_init_kernel(
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
    ops_arg arg6
)
{
#else
void ops_par_loop_tea_leaf_common_init_kernel_execute(ops_kernel_descriptor *desc)
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
#endif

//  ======
//  Timing
//  ======
    double __t1,__t2,__c1,__c2;

    ops_arg args[7];

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;
    args[3] = arg3;
    args[4] = arg4;
    args[5] = arg5;
    args[6] = arg6;

#if defined(CHECKPOINTING) && !defined(OPS_LAZY)
    if (!ops_checkpointing_before(args,7, range, 27)) return;
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timing_realloc(block->instance, 27, "tea_leaf_common_init_kernel");
        block->instance->OPS_kernels[27].count++;
        ops_timers_core(&__c1, &__t1);
    }

#ifdef OPS_DEBUG
    ops_register_args(block->instance, args, "tea_leaf_common_init_kernel");
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
    if (compute_ranges(args, 7, block, range, start, end, arg_idx) < 0) return;
#endif


//  =====================================================
//  Initialize global variable with the dimension of dats
//  =====================================================
    int xdim0_tea_leaf_common_init_kernel = args[0].dat->size[0];
    int xdim1_tea_leaf_common_init_kernel = args[1].dat->size[0];
    int xdim2_tea_leaf_common_init_kernel = args[2].dat->size[0];
    int xdim3_tea_leaf_common_init_kernel = args[3].dat->size[0];
    int xdim4_tea_leaf_common_init_kernel = args[4].dat->size[0];

//  =======================================================
//  Set up initial pointers and exchange halos if necessary
//  =======================================================
    int base0 = args[0].dat->base_offset;
    double * __restrict__ w_p = (double *)(args[0].data_d + base0);
    int base1 = args[1].dat->base_offset;
    double * __restrict__ r_p = (double *)(args[1].data_d + base1);
    int base2 = args[2].dat->base_offset;
    double * __restrict__ Kx_p = (double *)(args[2].data_d + base2);
    int base3 = args[3].dat->base_offset;
    double * __restrict__ Ky_p = (double *)(args[3].data_d + base3);
    int base4 = args[4].dat->base_offset;
    double * __restrict__ u_p = (double *)(args[4].data_d + base4);
    double rx_val = *(double *)args[5].data;

    double ry_val = *(double *)args[6].data;


#ifndef OPS_LAZY
//  ==============
//  Halo Exchanges
//  ==============
    ops_H_D_exchanges_device(args, 7);
    ops_halo_exchanges(args,7,range);
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[27].mpi_time += __t2 - __t1;
    }

    int start_0 = start[0];
    int end_0 = end[0];
    int start_1 = start[1];
    int end_1 = end[1];
    if ((end[0]-start[0])>0 && (end[1]-start[1])>0) {
        block->instance->sycl_instance->queue->submit([&](cl::sycl::handler &cgh) {


          cgh.parallel_for<class tea_leaf_common_init_kernel_kernel>(cl::sycl::nd_range<2>(cl::sycl::range<2>(
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

              ACC<double> w(xdim0_tea_leaf_common_init_kernel, w_p + n_x*1 + n_y * xdim0_tea_leaf_common_init_kernel*1);
              ACC<double> r(xdim1_tea_leaf_common_init_kernel, r_p + n_x*1 + n_y * xdim1_tea_leaf_common_init_kernel*1);
              const ACC<double> Kx(xdim2_tea_leaf_common_init_kernel, Kx_p + n_x*1 + n_y * xdim2_tea_leaf_common_init_kernel*1);
              const ACC<double> Ky(xdim3_tea_leaf_common_init_kernel, Ky_p + n_x*1 + n_y * xdim3_tea_leaf_common_init_kernel*1);
              const ACC<double> u(xdim4_tea_leaf_common_init_kernel, u_p + n_x*1 + n_y * xdim4_tea_leaf_common_init_kernel*1);
              const double *rx = &rx_val;
              const double *ry = &ry_val;
//  =========
//  User code
//  =========
                if (n_x < end_0 && n_y < end_1) {
                    
  w(0,0) = (1.0
        + (*ry)*(Ky(0, 1) + Ky(0,0))
        + (*rx)*(Kx(1, 0) + Kx(0,0)))*u(0,0)
        - (*ry)*(Ky(0, 1) *u(0, 1) + Ky(0,0)*u(0, -1))
        - (*rx)*(Kx(1, 0) *u(1, 0) + Kx(0,0)*u(-1, 0));
    r(0,0) = u(0,0) - w(0,0);

                }
            });
        });
    }

    if (block->instance->OPS_diags > 1) {
        block->instance->sycl_instance->queue->wait();
        ops_timers_core(&__c1, &__t1);
        block->instance->OPS_kernels[27].time += __t1 -__t2;
    }

#ifndef OPS_LAZY
    ops_set_dirtybit_device(args, 7);
    ops_set_halo_dirtybit3(&args[0], range);
    ops_set_halo_dirtybit3(&args[1], range);
#endif

    if (block->instance->OPS_diags > 1) {
//  ====================
//  Update kernel record
//  ====================
    ops_timers_core(&__c2, &__t2);
    block->instance->OPS_kernels[27].mpi_time += __t2 -__t1;
    block->instance->OPS_kernels[27].transfer += ops_compute_transfer(dim, start, end, &arg0);
    block->instance->OPS_kernels[27].transfer += ops_compute_transfer(dim, start, end, &arg1);
    block->instance->OPS_kernels[27].transfer += ops_compute_transfer(dim, start, end, &arg2);
    block->instance->OPS_kernels[27].transfer += ops_compute_transfer(dim, start, end, &arg3);
    block->instance->OPS_kernels[27].transfer += ops_compute_transfer(dim, start, end, &arg4);
}
}

#ifdef OPS_LAZY
void ops_par_loop_tea_leaf_common_init_kernel(
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
    ops_arg arg6
  )
{
    ops_arg args[7];

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;
    args[3] = arg3;
    args[4] = arg4;
    args[5] = arg5;
    args[6] = arg6;

    create_kerneldesc_and_enque("tea_leaf_common_init_kernel", args, 7, 27, dim, 1, range, block, ops_par_loop_tea_leaf_common_init_kernel_execute);
}
#endif
