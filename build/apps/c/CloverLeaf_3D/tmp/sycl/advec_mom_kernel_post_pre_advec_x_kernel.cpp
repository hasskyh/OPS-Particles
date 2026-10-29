// Auto-generated at 2026-10-29 00:14:06.121855 by ops-translator legacy


//  ==================
//  Host stub function
//  ==================
#ifndef OPS_LAZY
void ops_par_loop_advec_mom_kernel_post_pre_advec_x(
    const char *name,
    ops_block block,
    int dim,
    int* range,
    ops_arg arg0, 
    ops_arg arg1, 
    ops_arg arg2, 
    ops_arg arg3, 
    ops_arg arg4
)
{
#else
void ops_par_loop_advec_mom_kernel_post_pre_advec_x_execute(ops_kernel_descriptor *desc)
{
    ops_block block = desc->block;
    int dim = desc->dim;
    int *range = desc->range;
    ops_arg arg0 = desc->args[0];
    ops_arg arg1 = desc->args[1];
    ops_arg arg2 = desc->args[2];
    ops_arg arg3 = desc->args[3];
    ops_arg arg4 = desc->args[4];
#endif

//  ======
//  Timing
//  ======
    double __t1,__t2,__c1,__c2;

    ops_arg args[5];

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;
    args[3] = arg3;
    args[4] = arg4;

#if defined(CHECKPOINTING) && !defined(OPS_LAZY)
    if (!ops_checkpointing_before(args,5, range, 22)) return;
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timing_realloc(block->instance, 22, "advec_mom_kernel_post_pre_advec_x");
        block->instance->OPS_kernels[22].count++;
        ops_timers_core(&__c1, &__t1);
    }

#ifdef OPS_DEBUG
    ops_register_args(block->instance, args, "advec_mom_kernel_post_pre_advec_x");
#endif

//  =================================================
//  compute locally allocated range for the sub-block
//  =================================================
    int start[3];
    int end[3];
#if defined(OPS_MPI) && !defined(OPS_LAZY)
int arg_idx[3];
#endif
#if defined(OPS_LAZY) || !defined(OPS_MPI)
    for (int n = 0; n < 3; n++) {
        start[n] = range[2*n];
        end[n]   = range[2*n+1];
    }
#else
    if (compute_ranges(args, 5, block, range, start, end, arg_idx) < 0) return;
#endif


//  =====================================================
//  Initialize global variable with the dimension of dats
//  =====================================================
    int xdim0_advec_mom_kernel_post_pre_advec_x = args[0].dat->size[0];
    int ydim0_advec_mom_kernel_post_pre_advec_x = args[0].dat->size[1];
    int xdim1_advec_mom_kernel_post_pre_advec_x = args[1].dat->size[0];
    int ydim1_advec_mom_kernel_post_pre_advec_x = args[1].dat->size[1];
    int xdim2_advec_mom_kernel_post_pre_advec_x = args[2].dat->size[0];
    int ydim2_advec_mom_kernel_post_pre_advec_x = args[2].dat->size[1];
    int xdim3_advec_mom_kernel_post_pre_advec_x = args[3].dat->size[0];
    int ydim3_advec_mom_kernel_post_pre_advec_x = args[3].dat->size[1];
    int xdim4_advec_mom_kernel_post_pre_advec_x = args[4].dat->size[0];
    int ydim4_advec_mom_kernel_post_pre_advec_x = args[4].dat->size[1];

//  =======================================================
//  Set up initial pointers and exchange halos if necessary
//  =======================================================
    int base0 = args[0].dat->base_offset;
    double * __restrict__ node_mass_post_p = (double *)(args[0].data_d + base0);
    int base1 = args[1].dat->base_offset;
    double * __restrict__ post_vol_p = (double *)(args[1].data_d + base1);
    int base2 = args[2].dat->base_offset;
    double * __restrict__ density1_p = (double *)(args[2].data_d + base2);
    int base3 = args[3].dat->base_offset;
    double * __restrict__ node_mass_pre_p = (double *)(args[3].data_d + base3);
    int base4 = args[4].dat->base_offset;
    double * __restrict__ node_flux_p = (double *)(args[4].data_d + base4);

#ifndef OPS_LAZY
//  ==============
//  Halo Exchanges
//  ==============
    ops_H_D_exchanges_device(args, 5);
    ops_halo_exchanges(args,5,range);
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[22].mpi_time += __t2 - __t1;
    }

    int start_0 = start[0];
    int end_0 = end[0];
    int start_1 = start[1];
    int end_1 = end[1];
    int start_2 = start[2];
    int end_2 = end[2];
    if ((end[0]-start[0])>0 && (end[1]-start[1])>0 && (end[2]-start[2])>0) {
        block->instance->sycl_instance->queue->submit([&](cl::sycl::handler &cgh) {


          cgh.parallel_for<class advec_mom_kernel_post_pre_advec_x_kernel>(cl::sycl::nd_range<3>(cl::sycl::range<3>(
               ((end[2]-start[2]-1)/block->instance->OPS_block_size_z+1)*block->instance->OPS_block_size_z,
               ((end[1]-start[1]-1)/block->instance->OPS_block_size_y+1)*block->instance->OPS_block_size_y,
                ((end[0]-start[0]-1)/block->instance->OPS_block_size_x+1)*block->instance->OPS_block_size_x
                 ),cl::sycl::range<3>(
                 block->instance->OPS_block_size_z,
                 block->instance->OPS_block_size_y,
          block->instance->OPS_block_size_x
                 ))
          , [=](cl::sycl::nd_item<3> item
          ) [[intel::kernel_args_restrict]] {

              int n_z = item.get_global_id()[0]+start_2;
              int n_y = item.get_global_id()[1]+start_1;
              int n_x = item.get_global_id()[2]+start_0;

              ACC<double> node_mass_post(xdim0_advec_mom_kernel_post_pre_advec_x, ydim0_advec_mom_kernel_post_pre_advec_x, node_mass_post_p + n_x*1 + n_y * xdim0_advec_mom_kernel_post_pre_advec_x*1 + n_z * xdim0_advec_mom_kernel_post_pre_advec_x * ydim0_advec_mom_kernel_post_pre_advec_x*1);
              const ACC<double> post_vol(xdim1_advec_mom_kernel_post_pre_advec_x, ydim1_advec_mom_kernel_post_pre_advec_x, post_vol_p + n_x*1 + n_y * xdim1_advec_mom_kernel_post_pre_advec_x*1 + n_z * xdim1_advec_mom_kernel_post_pre_advec_x * ydim1_advec_mom_kernel_post_pre_advec_x*1);
              const ACC<double> density1(xdim2_advec_mom_kernel_post_pre_advec_x, ydim2_advec_mom_kernel_post_pre_advec_x, density1_p + n_x*1 + n_y * xdim2_advec_mom_kernel_post_pre_advec_x*1 + n_z * xdim2_advec_mom_kernel_post_pre_advec_x * ydim2_advec_mom_kernel_post_pre_advec_x*1);
              ACC<double> node_mass_pre(xdim3_advec_mom_kernel_post_pre_advec_x, ydim3_advec_mom_kernel_post_pre_advec_x, node_mass_pre_p + n_x*1 + n_y * xdim3_advec_mom_kernel_post_pre_advec_x*1 + n_z * xdim3_advec_mom_kernel_post_pre_advec_x * ydim3_advec_mom_kernel_post_pre_advec_x*1);
              const ACC<double> node_flux(xdim4_advec_mom_kernel_post_pre_advec_x, ydim4_advec_mom_kernel_post_pre_advec_x, node_flux_p + n_x*1 + n_y * xdim4_advec_mom_kernel_post_pre_advec_x*1 + n_z * xdim4_advec_mom_kernel_post_pre_advec_x * ydim4_advec_mom_kernel_post_pre_advec_x*1);
//  =========
//  User code
//  =========
                if (n_x < end_0 && n_y < end_1 && n_z < end_2) {
                    

  node_mass_post(0,0,0) = 0.125 * ( density1(0,-1,0) * post_vol(0,-1,0) +
                                              density1(0,0,0)   * post_vol(0,0,0)   +
                                              density1(-1,-1,0) * post_vol(-1,-1,0) +
                                              density1(-1,0,0)  * post_vol(-1,0,0) +
                                              density1(0,-1,-1) * post_vol(0,-1,-1) +
                                              density1(0,0,-1)   * post_vol(0,0,-1)   +
                                              density1(-1,-1,-1) * post_vol(-1,-1,-1) +
                                              density1(-1,0,-1)  * post_vol(-1,0,-1)  );

  node_mass_pre(0,0,0) = node_mass_post(0,0,0) - node_flux(-1,0,0) + node_flux(0,0,0);


                }
            });
        });
    }

    if (block->instance->OPS_diags > 1) {
        block->instance->sycl_instance->queue->wait();
        ops_timers_core(&__c1, &__t1);
        block->instance->OPS_kernels[22].time += __t1 -__t2;
    }

#ifndef OPS_LAZY
    ops_set_dirtybit_device(args, 5);
    ops_set_halo_dirtybit3(&args[0], range);
    ops_set_halo_dirtybit3(&args[3], range);
#endif

    if (block->instance->OPS_diags > 1) {
//  ====================
//  Update kernel record
//  ====================
    ops_timers_core(&__c2, &__t2);
    block->instance->OPS_kernels[22].mpi_time += __t2 -__t1;
    block->instance->OPS_kernels[22].transfer += ops_compute_transfer(dim, start, end, &arg0);
    block->instance->OPS_kernels[22].transfer += ops_compute_transfer(dim, start, end, &arg1);
    block->instance->OPS_kernels[22].transfer += ops_compute_transfer(dim, start, end, &arg2);
    block->instance->OPS_kernels[22].transfer += ops_compute_transfer(dim, start, end, &arg3);
    block->instance->OPS_kernels[22].transfer += ops_compute_transfer(dim, start, end, &arg4);
}
}

#ifdef OPS_LAZY
void ops_par_loop_advec_mom_kernel_post_pre_advec_x(
    const char *name,
    ops_block block,
    int dim,
    int* range,
    ops_arg arg0,
    ops_arg arg1,
    ops_arg arg2,
    ops_arg arg3,
    ops_arg arg4
  )
{
    ops_arg args[5];

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;
    args[3] = arg3;
    args[4] = arg4;

    create_kerneldesc_and_enque("advec_mom_kernel_post_pre_advec_x", args, 5, 22, dim, 1, range, block, ops_par_loop_advec_mom_kernel_post_pre_advec_x_execute);
}
#endif
