// Auto-generated at 2026-10-29 00:14:06.211433 by ops-translator legacy


//  ==================
//  Host stub function
//  ==================
#ifndef OPS_LAZY
void ops_par_loop_calc_dt_kernel_print(
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
void ops_par_loop_calc_dt_kernel_print_execute(ops_kernel_descriptor *desc)
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
    if (!ops_checkpointing_before(args,8, range, 36)) return;
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timing_realloc(block->instance, 36, "calc_dt_kernel_print");
        block->instance->OPS_kernels[36].count++;
        ops_timers_core(&__c1, &__t1);
    }

#ifdef OPS_DEBUG
    ops_register_args(block->instance, args, "calc_dt_kernel_print");
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
    if (compute_ranges(args, 8, block, range, start, end, arg_idx) < 0) return;
#endif


//  =====================================================
//  Initialize global variable with the dimension of dats
//  =====================================================
    int xdim0_calc_dt_kernel_print = args[0].dat->size[0];
    int ydim0_calc_dt_kernel_print = args[0].dat->size[1];
    int xdim1_calc_dt_kernel_print = args[1].dat->size[0];
    int ydim1_calc_dt_kernel_print = args[1].dat->size[1];
    int xdim2_calc_dt_kernel_print = args[2].dat->size[0];
    int ydim2_calc_dt_kernel_print = args[2].dat->size[1];
    int xdim3_calc_dt_kernel_print = args[3].dat->size[0];
    int ydim3_calc_dt_kernel_print = args[3].dat->size[1];
    int xdim4_calc_dt_kernel_print = args[4].dat->size[0];
    int ydim4_calc_dt_kernel_print = args[4].dat->size[1];
    int xdim5_calc_dt_kernel_print = args[5].dat->size[0];
    int ydim5_calc_dt_kernel_print = args[5].dat->size[1];
    int xdim6_calc_dt_kernel_print = args[6].dat->size[0];
    int ydim6_calc_dt_kernel_print = args[6].dat->size[1];

//  =======================================================
//  Set up initial pointers and exchange halos if necessary
//  =======================================================
    int base0 = args[0].dat->base_offset;
    double * __restrict__ xvel0_p = (double *)(args[0].data_d + base0);
    int base1 = args[1].dat->base_offset;
    double * __restrict__ yvel0_p = (double *)(args[1].data_d + base1);
    int base2 = args[2].dat->base_offset;
    double * __restrict__ zvel0_p = (double *)(args[2].data_d + base2);
    int base3 = args[3].dat->base_offset;
    double * __restrict__ density0_p = (double *)(args[3].data_d + base3);
    int base4 = args[4].dat->base_offset;
    double * __restrict__ energy0_p = (double *)(args[4].data_d + base4);
    int base5 = args[5].dat->base_offset;
    double * __restrict__ pressure_p = (double *)(args[5].data_d + base5);
    int base6 = args[6].dat->base_offset;
    double * __restrict__ soundspeed_p = (double *)(args[6].data_d + base6);
#ifdef OPS_MPI
    double * __restrict__ p_a7 = (double *)(((ops_reduction)args[7].data)->data + ((ops_reduction)args[7].data)->size * block->index);
#else //OPS_MPI
    double * __restrict__ p_a7 = (double *)((ops_reduction)args[7].data)->data;
#endif //OPS_MPI

    int maxblocks = (end[0]-start[0]-1)/block->instance->OPS_block_size_x+1;
    maxblocks *= (end[1]-start[1]-1)/block->instance->OPS_block_size_y+1;
    maxblocks *= (end[2]-start[2]-1)/block->instance->OPS_block_size_z+1;
    int reduct_bytes = 0;
    size_t reduct_size = 0;

    reduct_bytes += ROUND_UP(maxblocks*28*sizeof(double));
    reduct_size = MAX(reduct_size,28*sizeof(double));

    reallocReductArrays(block->instance,reduct_bytes);
    reduct_bytes = 0;

    arg7.data = block->instance->OPS_reduct_h + reduct_bytes;
    double *arg7_data_d = (double*)(block->instance->OPS_reduct_d + reduct_bytes);
    for (int b = 0; b < maxblocks; b++) {
        for (int d = 0; d < 28; d++) {
            ((double *)arg7.data)[d+b*28] = ZERO_double;
        }
    }
    reduct_bytes += ROUND_UP(maxblocks*28*sizeof(double));

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
        block->instance->OPS_kernels[36].mpi_time += __t2 - __t1;
    }

    int start_0 = start[0];
    int end_0 = end[0];
    int start_1 = start[1];
    int end_1 = end[1];
    int start_2 = start[2];
    int end_2 = end[2];
    if ((end[0]-start[0])>0 && (end[1]-start[1])>0 && (end[2]-start[2])>0) {
        block->instance->sycl_instance->queue->submit([&](cl::sycl::handler &cgh) {
          cl::sycl::accessor<char, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> local_mem(reduct_size * cl::sycl::range<1>(block->instance->OPS_block_size_x*block->instance->OPS_block_size_y*block->instance->OPS_block_size_z),cgh);


          cgh.parallel_for<class calc_dt_kernel_print_kernel>(cl::sycl::nd_range<3>(cl::sycl::range<3>(
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

              const ACC<double> xvel0(xdim0_calc_dt_kernel_print, ydim0_calc_dt_kernel_print, xvel0_p + n_x*1 + n_y * xdim0_calc_dt_kernel_print*1 + n_z * xdim0_calc_dt_kernel_print * ydim0_calc_dt_kernel_print*1);
              const ACC<double> yvel0(xdim1_calc_dt_kernel_print, ydim1_calc_dt_kernel_print, yvel0_p + n_x*1 + n_y * xdim1_calc_dt_kernel_print*1 + n_z * xdim1_calc_dt_kernel_print * ydim1_calc_dt_kernel_print*1);
              const ACC<double> zvel0(xdim2_calc_dt_kernel_print, ydim2_calc_dt_kernel_print, zvel0_p + n_x*1 + n_y * xdim2_calc_dt_kernel_print*1 + n_z * xdim2_calc_dt_kernel_print * ydim2_calc_dt_kernel_print*1);
              const ACC<double> density0(xdim3_calc_dt_kernel_print, ydim3_calc_dt_kernel_print, density0_p + n_x*1 + n_y * xdim3_calc_dt_kernel_print*1 + n_z * xdim3_calc_dt_kernel_print * ydim3_calc_dt_kernel_print*1);
              const ACC<double> energy0(xdim4_calc_dt_kernel_print, ydim4_calc_dt_kernel_print, energy0_p + n_x*1 + n_y * xdim4_calc_dt_kernel_print*1 + n_z * xdim4_calc_dt_kernel_print * ydim4_calc_dt_kernel_print*1);
              const ACC<double> pressure(xdim5_calc_dt_kernel_print, ydim5_calc_dt_kernel_print, pressure_p + n_x*1 + n_y * xdim5_calc_dt_kernel_print*1 + n_z * xdim5_calc_dt_kernel_print * ydim5_calc_dt_kernel_print*1);
              const ACC<double> soundspeed(xdim6_calc_dt_kernel_print, ydim6_calc_dt_kernel_print, soundspeed_p + n_x*1 + n_y * xdim6_calc_dt_kernel_print*1 + n_z * xdim6_calc_dt_kernel_print * ydim6_calc_dt_kernel_print*1);
              double output[28];
              output[0] = ZERO_double;
              output[1] = ZERO_double;
              output[2] = ZERO_double;
              output[3] = ZERO_double;
              output[4] = ZERO_double;
              output[5] = ZERO_double;
              output[6] = ZERO_double;
              output[7] = ZERO_double;
              output[8] = ZERO_double;
              output[9] = ZERO_double;
              output[10] = ZERO_double;
              output[11] = ZERO_double;
              output[12] = ZERO_double;
              output[13] = ZERO_double;
              output[14] = ZERO_double;
              output[15] = ZERO_double;
              output[16] = ZERO_double;
              output[17] = ZERO_double;
              output[18] = ZERO_double;
              output[19] = ZERO_double;
              output[20] = ZERO_double;
              output[21] = ZERO_double;
              output[22] = ZERO_double;
              output[23] = ZERO_double;
              output[24] = ZERO_double;
              output[25] = ZERO_double;
              output[26] = ZERO_double;
              output[27] = ZERO_double;
//  =========
//  User code
//  =========
                if (n_x < end_0 && n_y < end_1 && n_z < end_2) {
                    
  output[0] = xvel0(0,0,0);
  output[1] = yvel0(0,0,0);
  output[2] = zvel0(0,0,0);
  output[3] = xvel0(1,0,0);
  output[4] = yvel0(1,0,0);
  output[5] = zvel0(0,0,0);
  output[6] = xvel0(1,1,0);
  output[7] = yvel0(1,1,0);
  output[8] = zvel0(0,0,0);
  output[9] = xvel0(0,1,0);
  output[10] = yvel0(0,1,0);
  output[11] = zvel0(0,0,0);
  output[12] = xvel0(0,0,1);
  output[13] = yvel0(0,0,1);
  output[14] = zvel0(0,0,1);
  output[15] = xvel0(1,0,1);
  output[16] = yvel0(1,0,1);
  output[17] = zvel0(0,0,1);
  output[18] = xvel0(1,1,1);
  output[19] = yvel0(1,1,1);
  output[20] = zvel0(0,0,1);
  output[21] = xvel0(0,1,1);
  output[22] = yvel0(0,1,1);
  output[23] = zvel0(0,0,1);
  output[24] = density0(0,0,0);
  output[25] = energy0(0,0,0);
  output[26] = pressure(0,0,0);
  output[27] = soundspeed(0,0,0);


                }
                int group_size = item.get_local_range(0);
                group_size *= item.get_local_range(1);
                group_size *= item.get_local_range(2);
                for (int d = 0; d < 28; d++) {
                    ops_reduction_sycl<OPS_INC>(arg7_data_d + d+item.get_group_linear_id()*28, output[d], (double*)&local_mem[0], item, group_size);
                }
            });
        });
    }
    mvReductArraysToHost(block->instance,reduct_bytes);
    for (int b = 0; b < maxblocks; b++) {
        for (int d = 0; d < 28; d++) {
            p_a7[d] = p_a7[d] + ((double *)arg7.data)[d+b*28];
        }
    }

    if (block->instance->OPS_diags > 1) {
        block->instance->sycl_instance->queue->wait();
        ops_timers_core(&__c1, &__t1);
        block->instance->OPS_kernels[36].time += __t1 -__t2;
    }

#ifndef OPS_LAZY
    ops_set_dirtybit_device(args, 8);
#endif

    if (block->instance->OPS_diags > 1) {
//  ====================
//  Update kernel record
//  ====================
    ops_timers_core(&__c2, &__t2);
    block->instance->OPS_kernels[36].mpi_time += __t2 -__t1;
    block->instance->OPS_kernels[36].transfer += ops_compute_transfer(dim, start, end, &arg0);
    block->instance->OPS_kernels[36].transfer += ops_compute_transfer(dim, start, end, &arg1);
    block->instance->OPS_kernels[36].transfer += ops_compute_transfer(dim, start, end, &arg2);
    block->instance->OPS_kernels[36].transfer += ops_compute_transfer(dim, start, end, &arg3);
    block->instance->OPS_kernels[36].transfer += ops_compute_transfer(dim, start, end, &arg4);
    block->instance->OPS_kernels[36].transfer += ops_compute_transfer(dim, start, end, &arg5);
    block->instance->OPS_kernels[36].transfer += ops_compute_transfer(dim, start, end, &arg6);
}
}

#ifdef OPS_LAZY
void ops_par_loop_calc_dt_kernel_print(
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

    create_kerneldesc_and_enque("calc_dt_kernel_print", args, 8, 36, dim, 1, range, block, ops_par_loop_calc_dt_kernel_print_execute);
}
#endif
