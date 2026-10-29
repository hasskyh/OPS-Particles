// Auto-generated at 2026-10-29 00:14:10.626225 by ops-translator legacy


//  ==================
//  Host stub function
//  ==================
#ifndef OPS_LAZY
void ops_par_loop_initialise_chunk_kernel_x(
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
void ops_par_loop_initialise_chunk_kernel_x_execute(ops_kernel_descriptor *desc)
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
    if (!ops_checkpointing_before(args,3, range, 7)) return;
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timing_realloc(block->instance, 7, "initialise_chunk_kernel_x");
        block->instance->OPS_kernels[7].count++;
        ops_timers_core(&__c1, &__t1);
    }

#ifdef OPS_DEBUG
    ops_register_args(block->instance, args, "initialise_chunk_kernel_x");
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
    if (compute_ranges(args, 3, block, range, start, end, arg_idx) < 0) return;
#endif


//  =====================================================
//  Initialize global variable with the dimension of dats
//  =====================================================
    int xdim0_initialise_chunk_kernel_x = args[0].dat->size[0];
    int xdim1_initialise_chunk_kernel_x = args[1].dat->size[0];
    int xdim2_initialise_chunk_kernel_x = args[2].dat->size[0];

//  =======================================================
//  Set up initial pointers and exchange halos if necessary
//  =======================================================
    int base0 = args[0].dat->base_offset;
    double * __restrict__ vertexx_p = (double *)(args[0].data_d + base0);
    int base1 = args[1].dat->base_offset;
    int * __restrict__ xx_p = (int *)(args[1].data_d + base1);
    int base2 = args[2].dat->base_offset;
    double * __restrict__ vertexdx_p = (double *)(args[2].data_d + base2);

#ifndef OPS_LAZY
//  ==============
//  Halo Exchanges
//  ==============
    ops_H_D_exchanges_device(args, 3);
    ops_halo_exchanges(args,3,range);
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[7].mpi_time += __t2 - __t1;
    }

    int start_0 = start[0];
    int end_0 = end[0];
    int start_1 = start[1];
    int end_1 = end[1];
    if ((end[0]-start[0])>0 && (end[1]-start[1])>0) {
        block->instance->sycl_instance->queue->submit([&](cl::sycl::handler &cgh) {

          auto field_sycl = (*field_p).template get_access<cl::sycl::access::mode::read>(cgh);
          auto grid_sycl = (*grid_p).template get_access<cl::sycl::access::mode::read>(cgh);

          cgh.parallel_for<class initialise_chunk_kernel_x_kernel>(cl::sycl::nd_range<2>(cl::sycl::range<2>(
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

              ACC<double> vertexx(xdim0_initialise_chunk_kernel_x, vertexx_p + n_x*1 + n_y * xdim0_initialise_chunk_kernel_x*0);
              const ACC<int> xx(xdim1_initialise_chunk_kernel_x, xx_p + n_x*1 + n_y * xdim1_initialise_chunk_kernel_x*0);
              ACC<double> vertexdx(xdim2_initialise_chunk_kernel_x, vertexdx_p + n_x*1 + n_y * xdim2_initialise_chunk_kernel_x*0);
//  =========
//  User code
//  =========
                if (n_x < end_0 && n_y < end_1) {
                    

  int x_min=field_sycl[0].x_min-2;
  double min_x, d_x;

  d_x = (grid_sycl[0].xmax - grid_sycl[0].xmin)/(double)grid_sycl[0].x_cells;
  min_x=grid_sycl[0].xmin+d_x*field_sycl[0].left;

  vertexx(0,0) = min_x + d_x * (xx(0,0) - x_min);
  vertexdx(0,0) = (double)d_x;

                }
            });
        });
    }

    if (block->instance->OPS_diags > 1) {
        block->instance->sycl_instance->queue->wait();
        ops_timers_core(&__c1, &__t1);
        block->instance->OPS_kernels[7].time += __t1 -__t2;
    }

#ifndef OPS_LAZY
    ops_set_dirtybit_device(args, 3);
    ops_set_halo_dirtybit3(&args[0], range);
    ops_set_halo_dirtybit3(&args[2], range);
#endif

    if (block->instance->OPS_diags > 1) {
//  ====================
//  Update kernel record
//  ====================
    ops_timers_core(&__c2, &__t2);
    block->instance->OPS_kernels[7].mpi_time += __t2 -__t1;
    block->instance->OPS_kernels[7].transfer += ops_compute_transfer(dim, start, end, &arg0);
    block->instance->OPS_kernels[7].transfer += ops_compute_transfer(dim, start, end, &arg1);
    block->instance->OPS_kernels[7].transfer += ops_compute_transfer(dim, start, end, &arg2);
}
}

#ifdef OPS_LAZY
void ops_par_loop_initialise_chunk_kernel_x(
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

    create_kerneldesc_and_enque("initialise_chunk_kernel_x", args, 3, 7, dim, 1, range, block, ops_par_loop_initialise_chunk_kernel_x_execute);
}
#endif
