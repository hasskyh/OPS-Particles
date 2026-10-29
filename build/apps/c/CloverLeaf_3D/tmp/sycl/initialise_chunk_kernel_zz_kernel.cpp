// Auto-generated at 2026-10-29 00:14:06.273516 by ops-translator legacy


//  ==================
//  Host stub function
//  ==================
#ifndef OPS_LAZY
void ops_par_loop_initialise_chunk_kernel_zz(
    const char *name,
    ops_block block,
    int dim,
    int* range,
    ops_arg arg0, 
    ops_arg arg1
)
{
#else
void ops_par_loop_initialise_chunk_kernel_zz_execute(ops_kernel_descriptor *desc)
{
    ops_block block = desc->block;
    int dim = desc->dim;
    int *range = desc->range;
    ops_arg arg0 = desc->args[0];
    ops_arg arg1 = desc->args[1];
#endif

//  ======
//  Timing
//  ======
    double __t1,__t2,__c1,__c2;

    ops_arg args[2];

    args[0] = arg0;
    args[1] = arg1;

#if defined(CHECKPOINTING) && !defined(OPS_LAZY)
    if (!ops_checkpointing_before(args,2, range, 45)) return;
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timing_realloc(block->instance, 45, "initialise_chunk_kernel_zz");
        block->instance->OPS_kernels[45].count++;
        ops_timers_core(&__c1, &__t1);
    }

#ifdef OPS_DEBUG
    ops_register_args(block->instance, args, "initialise_chunk_kernel_zz");
#endif

//  =================================================
//  compute locally allocated range for the sub-block
//  =================================================
    int start[3];
    int end[3];
    int arg_idx[3];
#if defined(OPS_LAZY) || !defined(OPS_MPI)
    for (int n = 0; n < 3; n++) {
        start[n] = range[2*n];
        end[n]   = range[2*n+1];
    }
#else
    if (compute_ranges(args, 2, block, range, start, end, arg_idx) < 0) return;
#endif

#if defined(OPS_MPI)
#if defined(OPS_LAZY)
    sub_block_list sb = OPS_sub_block_list[block->index];
    arg_idx[0] = sb->decomp_disp[0];
    arg_idx[1] = sb->decomp_disp[1];
    arg_idx[2] = sb->decomp_disp[2];
#else
    arg_idx[0] -= start[0];
    arg_idx[1] -= start[1];
    arg_idx[2] -= start[2];
#endif
#else //OPS_MPI
    arg_idx[0] = 0;
    arg_idx[1] = 0;
    arg_idx[2] = 0;
#endif //OPS_MPI

//  =====================================================
//  Initialize global variable with the dimension of dats
//  =====================================================
    int xdim0_initialise_chunk_kernel_zz = args[0].dat->size[0];
    int ydim0_initialise_chunk_kernel_zz = args[0].dat->size[1];

//  =======================================================
//  Set up initial pointers and exchange halos if necessary
//  =======================================================
    int base0 = args[0].dat->base_offset;
    int * __restrict__ zz_p = (int *)(args[0].data_d + base0);

#ifndef OPS_LAZY
//  ==============
//  Halo Exchanges
//  ==============
    ops_H_D_exchanges_device(args, 2);
    ops_halo_exchanges(args,2,range);
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[45].mpi_time += __t2 - __t1;
    }

    int start_0 = start[0];
    int end_0 = end[0];
    int arg_idx_0 = arg_idx[0];
    int start_1 = start[1];
    int end_1 = end[1];
    int arg_idx_1 = arg_idx[1];
    int start_2 = start[2];
    int end_2 = end[2];
    int arg_idx_2 = arg_idx[2];
    if ((end[0]-start[0])>0 && (end[1]-start[1])>0 && (end[2]-start[2])>0) {
        block->instance->sycl_instance->queue->submit([&](cl::sycl::handler &cgh) {


          cgh.parallel_for<class initialise_chunk_kernel_zz_kernel>(cl::sycl::nd_range<3>(cl::sycl::range<3>(
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

              int idx[] = {arg_idx_0+n_x, arg_idx_1+n_y, arg_idx_2+n_z};

              ACC<int> zz(xdim0_initialise_chunk_kernel_zz, ydim0_initialise_chunk_kernel_zz, zz_p + n_x*0 + n_y * xdim0_initialise_chunk_kernel_zz*0 + n_z * xdim0_initialise_chunk_kernel_zz * ydim0_initialise_chunk_kernel_zz*1);
//  =========
//  User code
//  =========
                if (n_x < end_0 && n_y < end_1 && n_z < end_2) {
                    
  zz(0,0,0) = idx[2]-2;

                }
            });
        });
    }

    if (block->instance->OPS_diags > 1) {
        block->instance->sycl_instance->queue->wait();
        ops_timers_core(&__c1, &__t1);
        block->instance->OPS_kernels[45].time += __t1 -__t2;
    }

#ifndef OPS_LAZY
    ops_set_dirtybit_device(args, 2);
    ops_set_halo_dirtybit3(&args[0], range);
#endif

    if (block->instance->OPS_diags > 1) {
//  ====================
//  Update kernel record
//  ====================
    ops_timers_core(&__c2, &__t2);
    block->instance->OPS_kernels[45].mpi_time += __t2 -__t1;
    block->instance->OPS_kernels[45].transfer += ops_compute_transfer(dim, start, end, &arg0);
}
}

#ifdef OPS_LAZY
void ops_par_loop_initialise_chunk_kernel_zz(
    const char *name,
    ops_block block,
    int dim,
    int* range,
    ops_arg arg0,
    ops_arg arg1
  )
{
    ops_arg args[2];

    args[0] = arg0;
    args[1] = arg1;

    create_kerneldesc_and_enque("initialise_chunk_kernel_zz", args, 2, 45, dim, 1, range, block, ops_par_loop_initialise_chunk_kernel_zz_execute);
}
#endif
