// Auto-generated at 2026-10-29 00:14:06.822675 by ops-translator legacy


//  ==================
//  Host stub function
//  ==================
#ifndef OPS_LAZY
void ops_par_loop_update_halo_kernel2_xvel_minus_4_left(
    const char *name,
    ops_block block,
    int dim,
    int* range,
    ops_arg arg0, 
    ops_arg arg1, 
    ops_arg arg2
)
{
#else
void ops_par_loop_update_halo_kernel2_xvel_minus_4_left_execute(ops_kernel_descriptor *desc)
{
    ops_block block = desc->block;
    int dim = desc->dim;
    int *range = desc->range;
    ops_arg arg0 = desc->args[0];
    ops_arg arg1 = desc->args[1];
    ops_arg arg2 = desc->args[2];
#endif

//  ======
//  Timing
//  ======
    double __t1,__t2,__c1,__c2;

    ops_arg args[3];

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;

#if defined(CHECKPOINTING) && !defined(OPS_LAZY)
    if (!ops_checkpointing_before(args,3, range, 72)) return;
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timing_realloc(block->instance, 72, "update_halo_kernel2_xvel_minus_4_left");
        block->instance->OPS_kernels[72].count++;
        ops_timers_core(&__c1, &__t1);
    }

#ifdef OPS_DEBUG
    ops_register_args(block->instance, args, "update_halo_kernel2_xvel_minus_4_left");
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
    if (compute_ranges(args, 3, block, range, start, end, arg_idx) < 0) return;
#endif


//  =====================================================
//  Initialize global variable with the dimension of dats
//  =====================================================
    int xdim0_update_halo_kernel2_xvel_minus_4_left = args[0].dat->size[0];
    int ydim0_update_halo_kernel2_xvel_minus_4_left = args[0].dat->size[1];
    int xdim1_update_halo_kernel2_xvel_minus_4_left = args[1].dat->size[0];
    int ydim1_update_halo_kernel2_xvel_minus_4_left = args[1].dat->size[1];

//  =======================================================
//  Set up initial pointers and exchange halos if necessary
//  =======================================================
    int base0 = args[0].dat->base_offset;
    double * __restrict__ xvel0_p = (double *)(args[0].data_d + base0);
    int base1 = args[1].dat->base_offset;
    double * __restrict__ xvel1_p = (double *)(args[1].data_d + base1);
    int *arg2h = (int *)args[2].data;

    int consts_bytes = 0;

    consts_bytes += ROUND_UP(NUM_FIELDS*sizeof(int));
    reallocConstArrays(block->instance,consts_bytes);
    consts_bytes = 0;
    consts_bytes = 0;
    arg2.data = block->instance->OPS_consts_h + consts_bytes;
    int* arg2_data_d = (int*)( block->instance->OPS_consts_d + consts_bytes);
    for (int d=0; d<NUM_FIELDS; d++) ((int *)arg2.data)[d] = arg2h[d];
    consts_bytes += ROUND_UP(NUM_FIELDS*sizeof(int));
    mvConstArraysToDevice(block->instance,consts_bytes);

#ifndef OPS_LAZY
//  ==============
//  Halo Exchanges
//  ==============
    ops_H_D_exchanges_device(args, 3);
    ops_halo_exchanges(args,3,range);
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[72].mpi_time += __t2 - __t1;
    }

    int start_0 = start[0];
    int end_0 = end[0];
    int start_1 = start[1];
    int end_1 = end[1];
    int start_2 = start[2];
    int end_2 = end[2];
    if ((end[0]-start[0])>0 && (end[1]-start[1])>0 && (end[2]-start[2])>0) {
        block->instance->sycl_instance->queue->submit([&](cl::sycl::handler &cgh) {


          cgh.parallel_for<class update_halo_kernel2_xvel_minus_4_left_kernel>(cl::sycl::nd_range<3>(cl::sycl::range<3>(
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

              ACC<double> xvel0(xdim0_update_halo_kernel2_xvel_minus_4_left, ydim0_update_halo_kernel2_xvel_minus_4_left, xvel0_p + n_x*1 + n_y * xdim0_update_halo_kernel2_xvel_minus_4_left*1 + n_z * xdim0_update_halo_kernel2_xvel_minus_4_left * ydim0_update_halo_kernel2_xvel_minus_4_left*1);
              ACC<double> xvel1(xdim1_update_halo_kernel2_xvel_minus_4_left, ydim1_update_halo_kernel2_xvel_minus_4_left, xvel1_p + n_x*1 + n_y * xdim1_update_halo_kernel2_xvel_minus_4_left*1 + n_z * xdim1_update_halo_kernel2_xvel_minus_4_left * ydim1_update_halo_kernel2_xvel_minus_4_left*1);
              const int *fields = arg2_data_d;
//  =========
//  User code
//  =========
                if (n_x < end_0 && n_y < end_1 && n_z < end_2) {
                    
  if(fields[FIELD_XVEL0] == 1) xvel0(0,0,0) = -xvel0(4,0,0);
  if(fields[FIELD_XVEL1] == 1) xvel1(0,0,0) = -xvel1(4,0,0);

                }
            });
        });
    }

    if (block->instance->OPS_diags > 1) {
        block->instance->sycl_instance->queue->wait();
        ops_timers_core(&__c1, &__t1);
        block->instance->OPS_kernels[72].time += __t1 -__t2;
    }

#ifndef OPS_LAZY
    ops_set_dirtybit_device(args, 3);
    ops_set_halo_dirtybit3(&args[0], range);
    ops_set_halo_dirtybit3(&args[1], range);
#endif

    if (block->instance->OPS_diags > 1) {
//  ====================
//  Update kernel record
//  ====================
    ops_timers_core(&__c2, &__t2);
    block->instance->OPS_kernels[72].mpi_time += __t2 -__t1;
    block->instance->OPS_kernels[72].transfer += ops_compute_transfer(dim, start, end, &arg0);
    block->instance->OPS_kernels[72].transfer += ops_compute_transfer(dim, start, end, &arg1);
}
}

#ifdef OPS_LAZY
void ops_par_loop_update_halo_kernel2_xvel_minus_4_left(
    const char *name,
    ops_block block,
    int dim,
    int* range,
    ops_arg arg0,
    ops_arg arg1,
    ops_arg arg2
  )
{
    ops_arg args[3];

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;

    create_kerneldesc_and_enque("update_halo_kernel2_xvel_minus_4_left", args, 3, 72, dim, 1, range, block, ops_par_loop_update_halo_kernel2_xvel_minus_4_left_execute);
}
#endif
