// Auto-generated at 2026-10-29 00:13:52.374479 by ops-translator legacy


//  ==================
//  Host stub function
//  ==================
#ifndef OPS_LAZY
void ops_par_loop_advec_mom_kernel_y2(
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
void ops_par_loop_advec_mom_kernel_y2_execute(ops_kernel_descriptor *desc)
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
    if (!ops_checkpointing_before(args,4, range, 14)) return;
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timing_realloc(block->instance, 14, "advec_mom_kernel_y2");
        block->instance->OPS_kernels[14].count++;
        ops_timers_core(&__c1, &__t1);
    }

#ifdef OPS_DEBUG
    ops_register_args(block->instance, args, "advec_mom_kernel_y2");
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
    int xdim0_advec_mom_kernel_y2 = args[0].dat->size[0];
    int xdim1_advec_mom_kernel_y2 = args[1].dat->size[0];
    int xdim2_advec_mom_kernel_y2 = args[2].dat->size[0];
    int xdim3_advec_mom_kernel_y2 = args[3].dat->size[0];

//  =======================================================
//  Set up initial pointers and exchange halos if necessary
//  =======================================================
    int base0 = args[0].dat->base_offset;
    double * __restrict__ pre_vol_p = (double *)(args[0].data_d + base0);
    int base1 = args[1].dat->base_offset;
    double * __restrict__ post_vol_p = (double *)(args[1].data_d + base1);
    int base2 = args[2].dat->base_offset;
    double * __restrict__ volume_p = (double *)(args[2].data_d + base2);
    int base3 = args[3].dat->base_offset;
    double * __restrict__ vol_flux_x_p = (double *)(args[3].data_d + base3);

#ifndef OPS_LAZY
//  ==============
//  Halo Exchanges
//  ==============
    ops_H_D_exchanges_device(args, 4);
    ops_halo_exchanges(args,4,range);
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[14].mpi_time += __t2 - __t1;
    }

    int start_0 = start[0];
    int end_0 = end[0];
    int start_1 = start[1];
    int end_1 = end[1];
    if ((end[0]-start[0])>0 && (end[1]-start[1])>0) {
        block->instance->sycl_instance->queue->submit([&](cl::sycl::handler &cgh) {


          cgh.parallel_for<class advec_mom_kernel_y2_kernel>(cl::sycl::nd_range<2>(cl::sycl::range<2>(
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

              ACC<double> pre_vol(xdim0_advec_mom_kernel_y2, pre_vol_p + n_x*1 + n_y * xdim0_advec_mom_kernel_y2*1);
              ACC<double> post_vol(xdim1_advec_mom_kernel_y2, post_vol_p + n_x*1 + n_y * xdim1_advec_mom_kernel_y2*1);
              const ACC<double> volume(xdim2_advec_mom_kernel_y2, volume_p + n_x*1 + n_y * xdim2_advec_mom_kernel_y2*1);
              const ACC<double> vol_flux_x(xdim3_advec_mom_kernel_y2, vol_flux_x_p + n_x*1 + n_y * xdim3_advec_mom_kernel_y2*1);
//  =========
//  User code
//  =========
                if (n_x < end_0 && n_y < end_1) {
                    

  post_vol(0,0)  = volume(0,0) ;
  pre_vol(0,0)   = post_vol(0,0)  + vol_flux_x(1,0) - vol_flux_x(0,0);


                }
            });
        });
    }

    if (block->instance->OPS_diags > 1) {
        block->instance->sycl_instance->queue->wait();
        ops_timers_core(&__c1, &__t1);
        block->instance->OPS_kernels[14].time += __t1 -__t2;
    }

#ifndef OPS_LAZY
    ops_set_dirtybit_device(args, 4);
    ops_set_halo_dirtybit3(&args[0], range);
    ops_set_halo_dirtybit3(&args[1], range);
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
    block->instance->OPS_kernels[14].transfer += ops_compute_transfer(dim, start, end, &arg3);
}
}

#ifdef OPS_LAZY
void ops_par_loop_advec_mom_kernel_y2(
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

    create_kerneldesc_and_enque("advec_mom_kernel_y2", args, 4, 14, dim, 1, range, block, ops_par_loop_advec_mom_kernel_y2_execute);
}
#endif
