// Auto-generated at 2026-10-29 00:14:11.729649 by ops-translator legacy


//  ==================
//  Host stub function
//  ==================
#ifndef OPS_LAZY
void ops_par_loop_updateRK3_kernel(
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
    ops_arg arg7, 
    ops_arg arg8, 
    ops_arg arg9, 
    ops_arg arg10
)
{
#else
void ops_par_loop_updateRK3_kernel_execute(ops_kernel_descriptor *desc)
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
    ops_arg arg8 = desc->args[8];
    ops_arg arg9 = desc->args[9];
    ops_arg arg10 = desc->args[10];
#endif

//  ======
//  Timing
//  ======
    double __t1,__t2,__c1,__c2;

    ops_arg args[11];

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;
    args[3] = arg3;
    args[4] = arg4;
    args[5] = arg5;
    args[6] = arg6;
    args[7] = arg7;
    args[8] = arg8;
    args[9] = arg9;
    args[10] = arg10;

#if defined(CHECKPOINTING) && !defined(OPS_LAZY)
    if (!ops_checkpointing_before(args,11, range, 6)) return;
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timing_realloc(block->instance, 6, "updateRK3_kernel");
        block->instance->OPS_kernels[6].count++;
        ops_timers_core(&__c1, &__t1);
    }

#ifdef OPS_DEBUG
    ops_register_args(block->instance, args, "updateRK3_kernel");
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
    if (compute_ranges(args, 11, block, range, start, end, arg_idx) < 0) return;
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
    double * __restrict__ rhou_new_p = (double *)(args[1].data_d + base1);
    int base2 = args[2].dat->base_offset;
    double * __restrict__ rhoE_new_p = (double *)(args[2].data_d + base2);
    int base3 = args[3].dat->base_offset;
    double * __restrict__ rho_old_p = (double *)(args[3].data_d + base3);
    int base4 = args[4].dat->base_offset;
    double * __restrict__ rhou_old_p = (double *)(args[4].data_d + base4);
    int base5 = args[5].dat->base_offset;
    double * __restrict__ rhoE_old_p = (double *)(args[5].data_d + base5);
    int base6 = args[6].dat->base_offset;
    double * __restrict__ rho_res_p = (double *)(args[6].data_d + base6);
    int base7 = args[7].dat->base_offset;
    double * __restrict__ rhou_res_p = (double *)(args[7].data_d + base7);
    int base8 = args[8].dat->base_offset;
    double * __restrict__ rhoE_res_p = (double *)(args[8].data_d + base8);
    double a1_val = *(double *)args[9].data;

    double a2_val = *(double *)args[10].data;


#ifndef OPS_LAZY
//  ==============
//  Halo Exchanges
//  ==============
    ops_H_D_exchanges_device(args, 11);
    ops_halo_exchanges(args,11,range);
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[6].mpi_time += __t2 - __t1;
    }

    int start_0 = start[0];
    int end_0 = end[0];
    if ((end[0]-start[0])>0) {
        block->instance->sycl_instance->queue->submit([&](cl::sycl::handler &cgh) {

          auto dt_sycl = (*dt_p).template get_access<cl::sycl::access::mode::read>(cgh);

          cgh.parallel_for<class updateRK3_kernel_kernel>(cl::sycl::nd_range<1>(cl::sycl::range<1>(
                ((end[0]-start[0]-1)/block->instance->OPS_block_size_x+1)*block->instance->OPS_block_size_x
                 ),cl::sycl::range<1>(
          block->instance->OPS_block_size_x
                 ))
          , [=](cl::sycl::nd_item<1> item
          ) [[intel::kernel_args_restrict]] {

              int n_x = item.get_global_id()[0]+start_0;

              ACC<double> rho_new(rho_new_p + n_x*1);
              ACC<double> rhou_new(rhou_new_p + n_x*1);
              ACC<double> rhoE_new(rhoE_new_p + n_x*1);
              ACC<double> rho_old(rho_old_p + n_x*1);
              ACC<double> rhou_old(rhou_old_p + n_x*1);
              ACC<double> rhoE_old(rhoE_old_p + n_x*1);
              ACC<double> rho_res(rho_res_p + n_x*1);
              ACC<double> rhou_res(rhou_res_p + n_x*1);
              ACC<double> rhoE_res(rhoE_res_p + n_x*1);
              const double *a1 = &a1_val;
              const double *a2 = &a2_val;
//  =========
//  User code
//  =========
                if (n_x < end_0) {
                    

  rho_new(0) = rho_old(0) + dt_sycl[0] * a1[0] * (-rho_res(0));
  rhou_new(0) = rhou_old(0) + dt_sycl[0] * a1[0] * (-rhou_res(0));
  rhoE_new(0) = rhoE_old(0) + dt_sycl[0] * a1[0] * (-rhoE_res(0));

  rho_old(0) = rho_old(0) + dt_sycl[0] * a2[0] * (-rho_res(0));
  rhou_old(0) = rhou_old(0) + dt_sycl[0] * a2[0] * (-rhou_res(0));
  rhoE_old(0) = rhoE_old(0) + dt_sycl[0] * a2[0] * (-rhoE_res(0));
  rho_res(0) = 0;
  rhou_res(0) = 0;
  rhoE_res(0) = 0;
  
                }
            });
        });
    }

    if (block->instance->OPS_diags > 1) {
        block->instance->sycl_instance->queue->wait();
        ops_timers_core(&__c1, &__t1);
        block->instance->OPS_kernels[6].time += __t1 -__t2;
    }

#ifndef OPS_LAZY
    ops_set_dirtybit_device(args, 11);
    ops_set_halo_dirtybit3(&args[0], range);
    ops_set_halo_dirtybit3(&args[1], range);
    ops_set_halo_dirtybit3(&args[2], range);
    ops_set_halo_dirtybit3(&args[3], range);
    ops_set_halo_dirtybit3(&args[4], range);
    ops_set_halo_dirtybit3(&args[5], range);
    ops_set_halo_dirtybit3(&args[6], range);
    ops_set_halo_dirtybit3(&args[7], range);
    ops_set_halo_dirtybit3(&args[8], range);
#endif

    if (block->instance->OPS_diags > 1) {
//  ====================
//  Update kernel record
//  ====================
    ops_timers_core(&__c2, &__t2);
    block->instance->OPS_kernels[6].mpi_time += __t2 -__t1;
    block->instance->OPS_kernels[6].transfer += ops_compute_transfer(dim, start, end, &arg0);
    block->instance->OPS_kernels[6].transfer += ops_compute_transfer(dim, start, end, &arg1);
    block->instance->OPS_kernels[6].transfer += ops_compute_transfer(dim, start, end, &arg2);
    block->instance->OPS_kernels[6].transfer += ops_compute_transfer(dim, start, end, &arg3);
    block->instance->OPS_kernels[6].transfer += ops_compute_transfer(dim, start, end, &arg4);
    block->instance->OPS_kernels[6].transfer += ops_compute_transfer(dim, start, end, &arg5);
    block->instance->OPS_kernels[6].transfer += ops_compute_transfer(dim, start, end, &arg6);
    block->instance->OPS_kernels[6].transfer += ops_compute_transfer(dim, start, end, &arg7);
    block->instance->OPS_kernels[6].transfer += ops_compute_transfer(dim, start, end, &arg8);
}
}

#ifdef OPS_LAZY
void ops_par_loop_updateRK3_kernel(
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
    ops_arg arg7,
    ops_arg arg8,
    ops_arg arg9,
    ops_arg arg10
  )
{
    ops_arg args[11];

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;
    args[3] = arg3;
    args[4] = arg4;
    args[5] = arg5;
    args[6] = arg6;
    args[7] = arg7;
    args[8] = arg8;
    args[9] = arg9;
    args[10] = arg10;

    create_kerneldesc_and_enque("updateRK3_kernel", args, 11, 6, dim, 1, range, block, ops_par_loop_updateRK3_kernel_execute);
}
#endif
