// Auto-generated at 2026-10-29 00:13:52.318584 by ops-translator legacy


//  ==================
//  Host stub function
//  ==================
#ifndef OPS_LAZY
void ops_par_loop_advec_cell_kernel4_xdir(
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
void ops_par_loop_advec_cell_kernel4_xdir_execute(ops_kernel_descriptor *desc)
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
        ops_timing_realloc(block->instance, 6, "advec_cell_kernel4_xdir");
        block->instance->OPS_kernels[6].count++;
        ops_timers_core(&__c1, &__t1);
    }

#ifdef OPS_DEBUG
    ops_register_args(block->instance, args, "advec_cell_kernel4_xdir");
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
    if (compute_ranges(args, 11, block, range, start, end, arg_idx) < 0) return;
#endif


//  =====================================================
//  Initialize global variable with the dimension of dats
//  =====================================================
    int xdim0_advec_cell_kernel4_xdir = args[0].dat->size[0];
    int xdim1_advec_cell_kernel4_xdir = args[1].dat->size[0];
    int xdim2_advec_cell_kernel4_xdir = args[2].dat->size[0];
    int xdim3_advec_cell_kernel4_xdir = args[3].dat->size[0];
    int xdim4_advec_cell_kernel4_xdir = args[4].dat->size[0];
    int xdim5_advec_cell_kernel4_xdir = args[5].dat->size[0];
    int xdim6_advec_cell_kernel4_xdir = args[6].dat->size[0];
    int xdim7_advec_cell_kernel4_xdir = args[7].dat->size[0];
    int xdim8_advec_cell_kernel4_xdir = args[8].dat->size[0];
    int xdim9_advec_cell_kernel4_xdir = args[9].dat->size[0];
    int xdim10_advec_cell_kernel4_xdir = args[10].dat->size[0];

//  =======================================================
//  Set up initial pointers and exchange halos if necessary
//  =======================================================
    int base0 = args[0].dat->base_offset;
    double * __restrict__ density1_p = (double *)(args[0].data_d + base0);
    int base1 = args[1].dat->base_offset;
    double * __restrict__ energy1_p = (double *)(args[1].data_d + base1);
    int base2 = args[2].dat->base_offset;
    double * __restrict__ mass_flux_x_p = (double *)(args[2].data_d + base2);
    int base3 = args[3].dat->base_offset;
    double * __restrict__ vol_flux_x_p = (double *)(args[3].data_d + base3);
    int base4 = args[4].dat->base_offset;
    double * __restrict__ pre_vol_p = (double *)(args[4].data_d + base4);
    int base5 = args[5].dat->base_offset;
    double * __restrict__ post_vol_p = (double *)(args[5].data_d + base5);
    int base6 = args[6].dat->base_offset;
    double * __restrict__ pre_mass_p = (double *)(args[6].data_d + base6);
    int base7 = args[7].dat->base_offset;
    double * __restrict__ post_mass_p = (double *)(args[7].data_d + base7);
    int base8 = args[8].dat->base_offset;
    double * __restrict__ advec_vol_p = (double *)(args[8].data_d + base8);
    int base9 = args[9].dat->base_offset;
    double * __restrict__ post_ener_p = (double *)(args[9].data_d + base9);
    int base10 = args[10].dat->base_offset;
    double * __restrict__ ener_flux_p = (double *)(args[10].data_d + base10);

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
    int start_1 = start[1];
    int end_1 = end[1];
    if ((end[0]-start[0])>0 && (end[1]-start[1])>0) {
        block->instance->sycl_instance->queue->submit([&](cl::sycl::handler &cgh) {


          cgh.parallel_for<class advec_cell_kernel4_xdir_kernel>(cl::sycl::nd_range<2>(cl::sycl::range<2>(
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

              ACC<double> density1(xdim0_advec_cell_kernel4_xdir, density1_p + n_x*1 + n_y * xdim0_advec_cell_kernel4_xdir*1);
              ACC<double> energy1(xdim1_advec_cell_kernel4_xdir, energy1_p + n_x*1 + n_y * xdim1_advec_cell_kernel4_xdir*1);
              const ACC<double> mass_flux_x(xdim2_advec_cell_kernel4_xdir, mass_flux_x_p + n_x*1 + n_y * xdim2_advec_cell_kernel4_xdir*1);
              const ACC<double> vol_flux_x(xdim3_advec_cell_kernel4_xdir, vol_flux_x_p + n_x*1 + n_y * xdim3_advec_cell_kernel4_xdir*1);
              const ACC<double> pre_vol(xdim4_advec_cell_kernel4_xdir, pre_vol_p + n_x*1 + n_y * xdim4_advec_cell_kernel4_xdir*1);
              const ACC<double> post_vol(xdim5_advec_cell_kernel4_xdir, post_vol_p + n_x*1 + n_y * xdim5_advec_cell_kernel4_xdir*1);
              ACC<double> pre_mass(xdim6_advec_cell_kernel4_xdir, pre_mass_p + n_x*1 + n_y * xdim6_advec_cell_kernel4_xdir*1);
              ACC<double> post_mass(xdim7_advec_cell_kernel4_xdir, post_mass_p + n_x*1 + n_y * xdim7_advec_cell_kernel4_xdir*1);
              ACC<double> advec_vol(xdim8_advec_cell_kernel4_xdir, advec_vol_p + n_x*1 + n_y * xdim8_advec_cell_kernel4_xdir*1);
              ACC<double> post_ener(xdim9_advec_cell_kernel4_xdir, post_ener_p + n_x*1 + n_y * xdim9_advec_cell_kernel4_xdir*1);
              const ACC<double> ener_flux(xdim10_advec_cell_kernel4_xdir, ener_flux_p + n_x*1 + n_y * xdim10_advec_cell_kernel4_xdir*1);
//  =========
//  User code
//  =========
                if (n_x < end_0 && n_y < end_1) {
                    

  pre_mass(0,0) = density1(0,0) * pre_vol(0,0);
  post_mass(0,0) = pre_mass(0,0) + mass_flux_x(0,0) - mass_flux_x(1,0);
  post_ener(0,0) = ( energy1(0,0) * pre_mass(0,0) + ener_flux(0,0) - ener_flux(1,0))/post_mass(0,0);
  advec_vol(0,0) = pre_vol(0,0) + vol_flux_x(0,0) - vol_flux_x(1,0);
  density1(0,0) = post_mass(0,0)/advec_vol(0,0);
  energy1(0,0) = post_ener(0,0);


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
    ops_set_halo_dirtybit3(&args[6], range);
    ops_set_halo_dirtybit3(&args[7], range);
    ops_set_halo_dirtybit3(&args[8], range);
    ops_set_halo_dirtybit3(&args[9], range);
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
    block->instance->OPS_kernels[6].transfer += ops_compute_transfer(dim, start, end, &arg9);
    block->instance->OPS_kernels[6].transfer += ops_compute_transfer(dim, start, end, &arg10);
}
}

#ifdef OPS_LAZY
void ops_par_loop_advec_cell_kernel4_xdir(
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

    create_kerneldesc_and_enque("advec_cell_kernel4_xdir", args, 11, 6, dim, 1, range, block, ops_par_loop_advec_cell_kernel4_xdir_execute);
}
#endif
