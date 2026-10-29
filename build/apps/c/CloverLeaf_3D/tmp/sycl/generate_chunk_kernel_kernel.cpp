// Auto-generated at 2026-10-29 00:14:06.250310 by ops-translator legacy


//  ==================
//  Host stub function
//  ==================
#ifndef OPS_LAZY
void ops_par_loop_generate_chunk_kernel(
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
void ops_par_loop_generate_chunk_kernel_execute(ops_kernel_descriptor *desc)
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
    if (!ops_checkpointing_before(args,11, range, 41)) return;
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timing_realloc(block->instance, 41, "generate_chunk_kernel");
        block->instance->OPS_kernels[41].count++;
        ops_timers_core(&__c1, &__t1);
    }

#ifdef OPS_DEBUG
    ops_register_args(block->instance, args, "generate_chunk_kernel");
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
    if (compute_ranges(args, 11, block, range, start, end, arg_idx) < 0) return;
#endif


//  =====================================================
//  Initialize global variable with the dimension of dats
//  =====================================================
    int xdim0_generate_chunk_kernel = args[0].dat->size[0];
    int ydim0_generate_chunk_kernel = args[0].dat->size[1];
    int xdim1_generate_chunk_kernel = args[1].dat->size[0];
    int ydim1_generate_chunk_kernel = args[1].dat->size[1];
    int xdim2_generate_chunk_kernel = args[2].dat->size[0];
    int ydim2_generate_chunk_kernel = args[2].dat->size[1];
    int xdim3_generate_chunk_kernel = args[3].dat->size[0];
    int ydim3_generate_chunk_kernel = args[3].dat->size[1];
    int xdim4_generate_chunk_kernel = args[4].dat->size[0];
    int ydim4_generate_chunk_kernel = args[4].dat->size[1];
    int xdim5_generate_chunk_kernel = args[5].dat->size[0];
    int ydim5_generate_chunk_kernel = args[5].dat->size[1];
    int xdim6_generate_chunk_kernel = args[6].dat->size[0];
    int ydim6_generate_chunk_kernel = args[6].dat->size[1];
    int xdim7_generate_chunk_kernel = args[7].dat->size[0];
    int ydim7_generate_chunk_kernel = args[7].dat->size[1];
    int xdim8_generate_chunk_kernel = args[8].dat->size[0];
    int ydim8_generate_chunk_kernel = args[8].dat->size[1];
    int xdim9_generate_chunk_kernel = args[9].dat->size[0];
    int ydim9_generate_chunk_kernel = args[9].dat->size[1];
    int xdim10_generate_chunk_kernel = args[10].dat->size[0];
    int ydim10_generate_chunk_kernel = args[10].dat->size[1];

//  =======================================================
//  Set up initial pointers and exchange halos if necessary
//  =======================================================
    int base0 = args[0].dat->base_offset;
    double * __restrict__ vertexx_p = (double *)(args[0].data_d + base0);
    int base1 = args[1].dat->base_offset;
    double * __restrict__ vertexy_p = (double *)(args[1].data_d + base1);
    int base2 = args[2].dat->base_offset;
    double * __restrict__ vertexz_p = (double *)(args[2].data_d + base2);
    int base3 = args[3].dat->base_offset;
    double * __restrict__ energy0_p = (double *)(args[3].data_d + base3);
    int base4 = args[4].dat->base_offset;
    double * __restrict__ density0_p = (double *)(args[4].data_d + base4);
    int base5 = args[5].dat->base_offset;
    double * __restrict__ xvel0_p = (double *)(args[5].data_d + base5);
    int base6 = args[6].dat->base_offset;
    double * __restrict__ yvel0_p = (double *)(args[6].data_d + base6);
    int base7 = args[7].dat->base_offset;
    double * __restrict__ zvel0_p = (double *)(args[7].data_d + base7);
    int base8 = args[8].dat->base_offset;
    double * __restrict__ cellx_p = (double *)(args[8].data_d + base8);
    int base9 = args[9].dat->base_offset;
    double * __restrict__ celly_p = (double *)(args[9].data_d + base9);
    int base10 = args[10].dat->base_offset;
    double * __restrict__ cellz_p = (double *)(args[10].data_d + base10);

#ifndef OPS_LAZY
//  ==============
//  Halo Exchanges
//  ==============
    ops_H_D_exchanges_device(args, 11);
    ops_halo_exchanges(args,11,range);
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[41].mpi_time += __t2 - __t1;
    }

    int start_0 = start[0];
    int end_0 = end[0];
    int start_1 = start[1];
    int end_1 = end[1];
    int start_2 = start[2];
    int end_2 = end[2];
    if ((end[0]-start[0])>0 && (end[1]-start[1])>0 && (end[2]-start[2])>0) {
        block->instance->sycl_instance->queue->submit([&](cl::sycl::handler &cgh) {

          auto states_sycl = (*states_p).template get_access<cl::sycl::access::mode::read>(cgh);
          auto number_of_states_sycl = (*number_of_states_p).template get_access<cl::sycl::access::mode::read>(cgh);
          auto g_sphe_sycl = (*g_sphe_p).template get_access<cl::sycl::access::mode::read>(cgh);
          auto g_point_sycl = (*g_point_p).template get_access<cl::sycl::access::mode::read>(cgh);
          auto g_cube_sycl = (*g_cube_p).template get_access<cl::sycl::access::mode::read>(cgh);

          cgh.parallel_for<class generate_chunk_kernel_kernel>(cl::sycl::nd_range<3>(cl::sycl::range<3>(
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

              const ACC<double> vertexx(xdim0_generate_chunk_kernel, ydim0_generate_chunk_kernel, vertexx_p + n_x*1 + n_y * xdim0_generate_chunk_kernel*0 + n_z * xdim0_generate_chunk_kernel * ydim0_generate_chunk_kernel*0);
              const ACC<double> vertexy(xdim1_generate_chunk_kernel, ydim1_generate_chunk_kernel, vertexy_p + n_x*0 + n_y * xdim1_generate_chunk_kernel*1 + n_z * xdim1_generate_chunk_kernel * ydim1_generate_chunk_kernel*0);
              const ACC<double> vertexz(xdim2_generate_chunk_kernel, ydim2_generate_chunk_kernel, vertexz_p + n_x*0 + n_y * xdim2_generate_chunk_kernel*0 + n_z * xdim2_generate_chunk_kernel * ydim2_generate_chunk_kernel*1);
              ACC<double> energy0(xdim3_generate_chunk_kernel, ydim3_generate_chunk_kernel, energy0_p + n_x*1 + n_y * xdim3_generate_chunk_kernel*1 + n_z * xdim3_generate_chunk_kernel * ydim3_generate_chunk_kernel*1);
              ACC<double> density0(xdim4_generate_chunk_kernel, ydim4_generate_chunk_kernel, density0_p + n_x*1 + n_y * xdim4_generate_chunk_kernel*1 + n_z * xdim4_generate_chunk_kernel * ydim4_generate_chunk_kernel*1);
              ACC<double> xvel0(xdim5_generate_chunk_kernel, ydim5_generate_chunk_kernel, xvel0_p + n_x*1 + n_y * xdim5_generate_chunk_kernel*1 + n_z * xdim5_generate_chunk_kernel * ydim5_generate_chunk_kernel*1);
              ACC<double> yvel0(xdim6_generate_chunk_kernel, ydim6_generate_chunk_kernel, yvel0_p + n_x*1 + n_y * xdim6_generate_chunk_kernel*1 + n_z * xdim6_generate_chunk_kernel * ydim6_generate_chunk_kernel*1);
              ACC<double> zvel0(xdim7_generate_chunk_kernel, ydim7_generate_chunk_kernel, zvel0_p + n_x*1 + n_y * xdim7_generate_chunk_kernel*1 + n_z * xdim7_generate_chunk_kernel * ydim7_generate_chunk_kernel*1);
              const ACC<double> cellx(xdim8_generate_chunk_kernel, ydim8_generate_chunk_kernel, cellx_p + n_x*1 + n_y * xdim8_generate_chunk_kernel*0 + n_z * xdim8_generate_chunk_kernel * ydim8_generate_chunk_kernel*0);
              const ACC<double> celly(xdim9_generate_chunk_kernel, ydim9_generate_chunk_kernel, celly_p + n_x*0 + n_y * xdim9_generate_chunk_kernel*1 + n_z * xdim9_generate_chunk_kernel * ydim9_generate_chunk_kernel*0);
              const ACC<double> cellz(xdim10_generate_chunk_kernel, ydim10_generate_chunk_kernel, cellz_p + n_x*0 + n_y * xdim10_generate_chunk_kernel*0 + n_z * xdim10_generate_chunk_kernel * ydim10_generate_chunk_kernel*1);
//  =========
//  User code
//  =========
                if (n_x < end_0 && n_y < end_1 && n_z < end_2) {
                    

  double radius, x_cent, y_cent, z_cent;
  int is_in = 0;



  energy0(0,0,0)= states_sycl[0].energy;
  density0(0,0,0)= states_sycl[0].density;
  xvel0(0,0,0)=states_sycl[0].xvel;
  yvel0(0,0,0)=states_sycl[0].yvel;
  zvel0(0,0,0)=states_sycl[0].zvel;

  for(int i = 1; i<number_of_states_sycl[0]; i++) {

    x_cent=states_sycl[i].xmin;
    y_cent=states_sycl[i].ymin;
    z_cent=states_sycl[i].zmin;

    if (states_sycl[i].geometry == g_cube_sycl[0]) {
      for (int i1 = -1; i1 <= 0; i1++) {
        for (int j1 = -1; j1 <= 0; j1++) {
          for (int k1 = -1; k1 <= 0; k1++) {
            if(vertexx(1+i1,0,0) >= states_sycl[i].xmin  && vertexx(0+i1,0,0) < states_sycl[i].xmax) {
              if(vertexy(0,1+j1,0) >= states_sycl[i].ymin && vertexy(0,0+j1,0) < states_sycl[i].ymax) {
                if(vertexz(0,0,1+k1) >= states_sycl[i].zmin && vertexz(0,0,0+k1) < states_sycl[i].zmax) {
                  is_in=1;
                }
              }
            }
          }
        }
      }

      if(vertexx(1,0,0) >= states_sycl[i].xmin  && vertexx(0,0,0) < states_sycl[i].xmax) {
        if(vertexy(0,1,0) >= states_sycl[i].ymin && vertexy(0,0,0) < states_sycl[i].ymax) {
          if(vertexz(0,0,1) >= states_sycl[i].zmin && vertexz(0,0,0) < states_sycl[i].zmax) {
            energy0(0,0,0) = states_sycl[i].energy;
            density0(0,0,0) = states_sycl[i].density;
          }
        }
      }

      if (is_in) {
        xvel0(0,0,0) = states_sycl[i].xvel;
        yvel0(0,0,0) = states_sycl[i].yvel;
        zvel0(0,0,0) = states_sycl[i].zvel;
      }
    }
    else if(states_sycl[i].geometry == g_sphe_sycl[0]) {
      for (int i1 = -1; i1 <= 0; i1++) {
        for (int j1 = -1; j1 <= 0; j1++) {
          for (int k1 = -1; k1 <= 0; k1++) {
            radius = cl::sycl::sqrt ((cellx(0,0,0) - x_cent) * (cellx(0,0,0) - x_cent) +
                     (celly(0,0,0) - y_cent) * (celly(0,0,0) - y_cent) +
                     (cellz(0,0,0) - z_cent) * (cellz(0,0,0) - z_cent));
            if(radius <= states_sycl[i].radius) is_in = 1;
          }
        }
      }
      if(radius <= states_sycl[i].radius) {
        energy0(0,0,0) = states_sycl[i].energy;
        density0(0,0,0) = states_sycl[i].density;
      }
      if (is_in) {
        xvel0(0,0,0) = states_sycl[i].xvel;
        yvel0(0,0,0) = states_sycl[i].yvel;
        zvel0(0,0,0) = states_sycl[i].zvel;

      }
    }
    else if(states_sycl[i].geometry == g_point_sycl[0]) {
      for (int i1 = -1; i1 <= 0; i1++) {
        for (int j1 = -1; j1 <= 0; j1++) {
          for (int k1 = -1; k1 <= 0; k1++) {
            if(vertexx(0+i1,0,0) == x_cent && vertexy(0,0+j1,0) == y_cent && vertexz(0,0,0+k1) == z_cent)
              is_in = 1;
          }
        }
      }

      if(vertexx(0,0,0) == x_cent && vertexy(0,0,0) == y_cent && vertexz(0,0,0) == z_cent) {
        energy0(0,0,0) = states_sycl[i].energy;
        density0(0,0,0) = states_sycl[i].density;
      }
      if (is_in) {
        xvel0(0,0,0) = states_sycl[i].xvel;
        yvel0(0,0,0) = states_sycl[i].yvel;
        zvel0(0,0,0) = states_sycl[i].zvel;
      }
    }
  }

                }
            });
        });
    }

    if (block->instance->OPS_diags > 1) {
        block->instance->sycl_instance->queue->wait();
        ops_timers_core(&__c1, &__t1);
        block->instance->OPS_kernels[41].time += __t1 -__t2;
    }

#ifndef OPS_LAZY
    ops_set_dirtybit_device(args, 11);
    ops_set_halo_dirtybit3(&args[3], range);
    ops_set_halo_dirtybit3(&args[4], range);
    ops_set_halo_dirtybit3(&args[5], range);
    ops_set_halo_dirtybit3(&args[6], range);
    ops_set_halo_dirtybit3(&args[7], range);
#endif

    if (block->instance->OPS_diags > 1) {
//  ====================
//  Update kernel record
//  ====================
    ops_timers_core(&__c2, &__t2);
    block->instance->OPS_kernels[41].mpi_time += __t2 -__t1;
    block->instance->OPS_kernels[41].transfer += ops_compute_transfer(dim, start, end, &arg0);
    block->instance->OPS_kernels[41].transfer += ops_compute_transfer(dim, start, end, &arg1);
    block->instance->OPS_kernels[41].transfer += ops_compute_transfer(dim, start, end, &arg2);
    block->instance->OPS_kernels[41].transfer += ops_compute_transfer(dim, start, end, &arg3);
    block->instance->OPS_kernels[41].transfer += ops_compute_transfer(dim, start, end, &arg4);
    block->instance->OPS_kernels[41].transfer += ops_compute_transfer(dim, start, end, &arg5);
    block->instance->OPS_kernels[41].transfer += ops_compute_transfer(dim, start, end, &arg6);
    block->instance->OPS_kernels[41].transfer += ops_compute_transfer(dim, start, end, &arg7);
    block->instance->OPS_kernels[41].transfer += ops_compute_transfer(dim, start, end, &arg8);
    block->instance->OPS_kernels[41].transfer += ops_compute_transfer(dim, start, end, &arg9);
    block->instance->OPS_kernels[41].transfer += ops_compute_transfer(dim, start, end, &arg10);
}
}

#ifdef OPS_LAZY
void ops_par_loop_generate_chunk_kernel(
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

    create_kerneldesc_and_enque("generate_chunk_kernel", args, 11, 41, dim, 1, range, block, ops_par_loop_generate_chunk_kernel_execute);
}
#endif
