// Auto-generated at 2026-10-29 00:14:05.948337 by ops-translator legacy


//  ==================
//  Host stub function
//  ==================
#ifndef OPS_LAZY
void ops_par_loop_accelerate_kernel(
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
    ops_arg arg10, 
    ops_arg arg11, 
    ops_arg arg12, 
    ops_arg arg13
)
{
#else
void ops_par_loop_accelerate_kernel_execute(ops_kernel_descriptor *desc)
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
    ops_arg arg11 = desc->args[11];
    ops_arg arg12 = desc->args[12];
    ops_arg arg13 = desc->args[13];
#endif

//  ======
//  Timing
//  ======
    double __t1,__t2,__c1,__c2;

    ops_arg args[14];

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
    args[11] = arg11;
    args[12] = arg12;
    args[13] = arg13;

#if defined(CHECKPOINTING) && !defined(OPS_LAZY)
    if (!ops_checkpointing_before(args,14, range, 2)) return;
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timing_realloc(block->instance, 2, "accelerate_kernel");
        block->instance->OPS_kernels[2].count++;
        ops_timers_core(&__c1, &__t1);
    }

#ifdef OPS_DEBUG
    ops_register_args(block->instance, args, "accelerate_kernel");
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
    if (compute_ranges(args, 14, block, range, start, end, arg_idx) < 0) return;
#endif


//  =====================================================
//  Initialize global variable with the dimension of dats
//  =====================================================
    int xdim0_accelerate_kernel = args[0].dat->size[0];
    int ydim0_accelerate_kernel = args[0].dat->size[1];
    int xdim1_accelerate_kernel = args[1].dat->size[0];
    int ydim1_accelerate_kernel = args[1].dat->size[1];
    int xdim2_accelerate_kernel = args[2].dat->size[0];
    int ydim2_accelerate_kernel = args[2].dat->size[1];
    int xdim3_accelerate_kernel = args[3].dat->size[0];
    int ydim3_accelerate_kernel = args[3].dat->size[1];
    int xdim4_accelerate_kernel = args[4].dat->size[0];
    int ydim4_accelerate_kernel = args[4].dat->size[1];
    int xdim5_accelerate_kernel = args[5].dat->size[0];
    int ydim5_accelerate_kernel = args[5].dat->size[1];
    int xdim6_accelerate_kernel = args[6].dat->size[0];
    int ydim6_accelerate_kernel = args[6].dat->size[1];
    int xdim7_accelerate_kernel = args[7].dat->size[0];
    int ydim7_accelerate_kernel = args[7].dat->size[1];
    int xdim8_accelerate_kernel = args[8].dat->size[0];
    int ydim8_accelerate_kernel = args[8].dat->size[1];
    int xdim9_accelerate_kernel = args[9].dat->size[0];
    int ydim9_accelerate_kernel = args[9].dat->size[1];
    int xdim10_accelerate_kernel = args[10].dat->size[0];
    int ydim10_accelerate_kernel = args[10].dat->size[1];
    int xdim11_accelerate_kernel = args[11].dat->size[0];
    int ydim11_accelerate_kernel = args[11].dat->size[1];
    int xdim12_accelerate_kernel = args[12].dat->size[0];
    int ydim12_accelerate_kernel = args[12].dat->size[1];
    int xdim13_accelerate_kernel = args[13].dat->size[0];
    int ydim13_accelerate_kernel = args[13].dat->size[1];

//  =======================================================
//  Set up initial pointers and exchange halos if necessary
//  =======================================================
    int base0 = args[0].dat->base_offset;
    double * __restrict__ density0_p = (double *)(args[0].data_d + base0);
    int base1 = args[1].dat->base_offset;
    double * __restrict__ volume_p = (double *)(args[1].data_d + base1);
    int base2 = args[2].dat->base_offset;
    double * __restrict__ stepbymass_p = (double *)(args[2].data_d + base2);
    int base3 = args[3].dat->base_offset;
    double * __restrict__ xvel0_p = (double *)(args[3].data_d + base3);
    int base4 = args[4].dat->base_offset;
    double * __restrict__ xvel1_p = (double *)(args[4].data_d + base4);
    int base5 = args[5].dat->base_offset;
    double * __restrict__ xarea_p = (double *)(args[5].data_d + base5);
    int base6 = args[6].dat->base_offset;
    double * __restrict__ pressure_p = (double *)(args[6].data_d + base6);
    int base7 = args[7].dat->base_offset;
    double * __restrict__ yvel0_p = (double *)(args[7].data_d + base7);
    int base8 = args[8].dat->base_offset;
    double * __restrict__ yvel1_p = (double *)(args[8].data_d + base8);
    int base9 = args[9].dat->base_offset;
    double * __restrict__ yarea_p = (double *)(args[9].data_d + base9);
    int base10 = args[10].dat->base_offset;
    double * __restrict__ viscosity_p = (double *)(args[10].data_d + base10);
    int base11 = args[11].dat->base_offset;
    double * __restrict__ zvel0_p = (double *)(args[11].data_d + base11);
    int base12 = args[12].dat->base_offset;
    double * __restrict__ zvel1_p = (double *)(args[12].data_d + base12);
    int base13 = args[13].dat->base_offset;
    double * __restrict__ zarea_p = (double *)(args[13].data_d + base13);

#ifndef OPS_LAZY
//  ==============
//  Halo Exchanges
//  ==============
    ops_H_D_exchanges_device(args, 14);
    ops_halo_exchanges(args,14,range);
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[2].mpi_time += __t2 - __t1;
    }

    int start_0 = start[0];
    int end_0 = end[0];
    int start_1 = start[1];
    int end_1 = end[1];
    int start_2 = start[2];
    int end_2 = end[2];
    if ((end[0]-start[0])>0 && (end[1]-start[1])>0 && (end[2]-start[2])>0) {
        block->instance->sycl_instance->queue->submit([&](cl::sycl::handler &cgh) {

          auto dt_sycl = (*dt_p).template get_access<cl::sycl::access::mode::read>(cgh);

          cgh.parallel_for<class accelerate_kernel_kernel>(cl::sycl::nd_range<3>(cl::sycl::range<3>(
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

              const ACC<double> density0(xdim0_accelerate_kernel, ydim0_accelerate_kernel, density0_p + n_x*1 + n_y * xdim0_accelerate_kernel*1 + n_z * xdim0_accelerate_kernel * ydim0_accelerate_kernel*1);
              const ACC<double> volume(xdim1_accelerate_kernel, ydim1_accelerate_kernel, volume_p + n_x*1 + n_y * xdim1_accelerate_kernel*1 + n_z * xdim1_accelerate_kernel * ydim1_accelerate_kernel*1);
              ACC<double> stepbymass(xdim2_accelerate_kernel, ydim2_accelerate_kernel, stepbymass_p + n_x*1 + n_y * xdim2_accelerate_kernel*1 + n_z * xdim2_accelerate_kernel * ydim2_accelerate_kernel*1);
              const ACC<double> xvel0(xdim3_accelerate_kernel, ydim3_accelerate_kernel, xvel0_p + n_x*1 + n_y * xdim3_accelerate_kernel*1 + n_z * xdim3_accelerate_kernel * ydim3_accelerate_kernel*1);
              ACC<double> xvel1(xdim4_accelerate_kernel, ydim4_accelerate_kernel, xvel1_p + n_x*1 + n_y * xdim4_accelerate_kernel*1 + n_z * xdim4_accelerate_kernel * ydim4_accelerate_kernel*1);
              const ACC<double> xarea(xdim5_accelerate_kernel, ydim5_accelerate_kernel, xarea_p + n_x*1 + n_y * xdim5_accelerate_kernel*1 + n_z * xdim5_accelerate_kernel * ydim5_accelerate_kernel*1);
              const ACC<double> pressure(xdim6_accelerate_kernel, ydim6_accelerate_kernel, pressure_p + n_x*1 + n_y * xdim6_accelerate_kernel*1 + n_z * xdim6_accelerate_kernel * ydim6_accelerate_kernel*1);
              const ACC<double> yvel0(xdim7_accelerate_kernel, ydim7_accelerate_kernel, yvel0_p + n_x*1 + n_y * xdim7_accelerate_kernel*1 + n_z * xdim7_accelerate_kernel * ydim7_accelerate_kernel*1);
              ACC<double> yvel1(xdim8_accelerate_kernel, ydim8_accelerate_kernel, yvel1_p + n_x*1 + n_y * xdim8_accelerate_kernel*1 + n_z * xdim8_accelerate_kernel * ydim8_accelerate_kernel*1);
              const ACC<double> yarea(xdim9_accelerate_kernel, ydim9_accelerate_kernel, yarea_p + n_x*1 + n_y * xdim9_accelerate_kernel*1 + n_z * xdim9_accelerate_kernel * ydim9_accelerate_kernel*1);
              const ACC<double> viscosity(xdim10_accelerate_kernel, ydim10_accelerate_kernel, viscosity_p + n_x*1 + n_y * xdim10_accelerate_kernel*1 + n_z * xdim10_accelerate_kernel * ydim10_accelerate_kernel*1);
              const ACC<double> zvel0(xdim11_accelerate_kernel, ydim11_accelerate_kernel, zvel0_p + n_x*1 + n_y * xdim11_accelerate_kernel*1 + n_z * xdim11_accelerate_kernel * ydim11_accelerate_kernel*1);
              ACC<double> zvel1(xdim12_accelerate_kernel, ydim12_accelerate_kernel, zvel1_p + n_x*1 + n_y * xdim12_accelerate_kernel*1 + n_z * xdim12_accelerate_kernel * ydim12_accelerate_kernel*1);
              const ACC<double> zarea(xdim13_accelerate_kernel, ydim13_accelerate_kernel, zarea_p + n_x*1 + n_y * xdim13_accelerate_kernel*1 + n_z * xdim13_accelerate_kernel * ydim13_accelerate_kernel*1);
//  =========
//  User code
//  =========
                if (n_x < end_0 && n_y < end_1 && n_z < end_2) {
                    

  double nodal_mass = 0.0;
  nodal_mass =(density0(-1,-1, 0) * volume(-1,-1, 0) +
               density0( 0,-1, 0) * volume( 0,-1, 0) +
               density0( 0, 0, 0) * volume( 0, 0, 0) +
               density0(-1, 0, 0) * volume(-1, 0, 0) +
               density0(-1,-1,-1) * volume(-1,-1,-1) +
               density0( 0,-1,-1) * volume( 0,-1,-1) +
               density0( 0, 0,-1) * volume( 0, 0,-1) +
               density0(-1, 0,-1) * volume(-1, 0,-1)) * 0.125;

  stepbymass(0,0,0) = 0.25*dt_sycl[0] / nodal_mass;

  xvel1(0,0,0) = xvel0(0,0,0) - stepbymass(0,0,0) *
            ( xarea(0,0,0)  * ( pressure(0,0,0) - pressure(-1,0,0) ) +
              xarea(0,-1,0) * ( pressure(0,-1,0) - pressure(-1,-1,0) ) +
              xarea(0,0,-1) * ( pressure(0,0,-1) - pressure(-1,0,-1) ) +
              xarea(0,-1,-1) * ( pressure(0,-1,-1) - pressure(-1,-1,-1) ) );

  yvel1(0,0,0) = yvel0(0,0,0) - stepbymass(0,0,0) *
            ( yarea(0,0,0)  * ( pressure(0,0,0) - pressure(0,-1,0) ) +
              yarea(-1,0,0) * ( pressure(-1,0,0) - pressure(-1,-1,0) ) +
              yarea(0,0,-1) * ( pressure(0,0,-1) - pressure(0,-1,-1) ) +
              yarea(-1,0,-1)* ( pressure(-1,0,-1) - pressure(-1,-1,-1) ) );

  zvel1(0,0,0) = zvel0(0,0,0) - stepbymass(0,0,0) *
            ( zarea(0,0,0)  * ( pressure(0,0,0) - pressure(0,0,-1) ) +
              zarea(0,-1,0) * ( pressure(0,-1,0) - pressure(0,-1,-1) ) +
              zarea(-1,0,0) * ( pressure(-1,0,0) - pressure(-1,0,-1) ) +
              zarea(-1,-1,0)* ( pressure(-1,-1,0) - pressure(-1,-1,-1) ) );


  xvel1(0,0,0) = xvel1(0,0,0) - stepbymass(0,0,0) *
            ( xarea(0,0,0)  * ( viscosity(0,0,0) - viscosity(-1,0,0) ) +
              xarea(0,-1,0) * ( viscosity(0,-1,0) - viscosity(-1,-1,0) ) +
              xarea(0,0,-1) * ( viscosity(0,0,-1) - viscosity(-1,0,-1) ) +
              xarea(0,-1,-1)* ( viscosity(0,-1,-1) - viscosity(-1,-1,-1) ) );

  yvel1(0,0,0) = yvel1(0,0,0) - stepbymass(0,0,0) *
            ( yarea(0,0,0)  * ( viscosity(0,0,0)  - viscosity(0,-1,0) ) +
              yarea(-1,0,0) * ( viscosity(-1,0,0) - viscosity(-1,-1,0) ) +
              yarea(0,0,-1) * ( viscosity(0,0,-1) - viscosity(0,-1,-1) ) +
              yarea(-1,0,-1)* ( viscosity(-1,0,-1)- viscosity(-1,-1,-1) ) );

  zvel1(0,0,0) = zvel1(0,0,0) - stepbymass(0,0,0) *
            ( zarea(0,0,0)  * ( viscosity(0,0,0)  - viscosity(0,0,-1) ) +
              zarea(0,-1,0) * ( viscosity(0,-1,0) - viscosity(0,-1,-1) ) +
              zarea(-1,0,0) * ( viscosity(-1,0,0) - viscosity(-1,0,-1) ) +
              zarea(-1,-1,0)* ( viscosity(-1,-1,0)- viscosity(-1,-1,-1) ) );



                }
            });
        });
    }

    if (block->instance->OPS_diags > 1) {
        block->instance->sycl_instance->queue->wait();
        ops_timers_core(&__c1, &__t1);
        block->instance->OPS_kernels[2].time += __t1 -__t2;
    }

#ifndef OPS_LAZY
    ops_set_dirtybit_device(args, 14);
    ops_set_halo_dirtybit3(&args[2], range);
    ops_set_halo_dirtybit3(&args[4], range);
    ops_set_halo_dirtybit3(&args[8], range);
    ops_set_halo_dirtybit3(&args[12], range);
#endif

    if (block->instance->OPS_diags > 1) {
//  ====================
//  Update kernel record
//  ====================
    ops_timers_core(&__c2, &__t2);
    block->instance->OPS_kernels[2].mpi_time += __t2 -__t1;
    block->instance->OPS_kernels[2].transfer += ops_compute_transfer(dim, start, end, &arg0);
    block->instance->OPS_kernels[2].transfer += ops_compute_transfer(dim, start, end, &arg1);
    block->instance->OPS_kernels[2].transfer += ops_compute_transfer(dim, start, end, &arg2);
    block->instance->OPS_kernels[2].transfer += ops_compute_transfer(dim, start, end, &arg3);
    block->instance->OPS_kernels[2].transfer += ops_compute_transfer(dim, start, end, &arg4);
    block->instance->OPS_kernels[2].transfer += ops_compute_transfer(dim, start, end, &arg5);
    block->instance->OPS_kernels[2].transfer += ops_compute_transfer(dim, start, end, &arg6);
    block->instance->OPS_kernels[2].transfer += ops_compute_transfer(dim, start, end, &arg7);
    block->instance->OPS_kernels[2].transfer += ops_compute_transfer(dim, start, end, &arg8);
    block->instance->OPS_kernels[2].transfer += ops_compute_transfer(dim, start, end, &arg9);
    block->instance->OPS_kernels[2].transfer += ops_compute_transfer(dim, start, end, &arg10);
    block->instance->OPS_kernels[2].transfer += ops_compute_transfer(dim, start, end, &arg11);
    block->instance->OPS_kernels[2].transfer += ops_compute_transfer(dim, start, end, &arg12);
    block->instance->OPS_kernels[2].transfer += ops_compute_transfer(dim, start, end, &arg13);
}
}

#ifdef OPS_LAZY
void ops_par_loop_accelerate_kernel(
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
    ops_arg arg10,
    ops_arg arg11,
    ops_arg arg12,
    ops_arg arg13
  )
{
    ops_arg args[14];

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
    args[11] = arg11;
    args[12] = arg12;
    args[13] = arg13;

    create_kerneldesc_and_enque("accelerate_kernel", args, 14, 2, dim, 1, range, block, ops_par_loop_accelerate_kernel_execute);
}
#endif
