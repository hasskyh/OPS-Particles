// Auto-generated at 2026-10-29 00:14:06.307290 by ops-translator legacy


//  ==================
//  Host stub function
//  ==================
#ifndef OPS_LAZY
void ops_par_loop_initialise_chunk_kernel_celly(
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
void ops_par_loop_initialise_chunk_kernel_celly_execute(ops_kernel_descriptor *desc)
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
    if (!ops_checkpointing_before(args,3, range, 50)) return;
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timing_realloc(block->instance, 50, "initialise_chunk_kernel_celly");
        block->instance->OPS_kernels[50].count++;
        ops_timers_core(&__c1, &__t1);
    }

#ifdef OPS_DEBUG
    ops_register_args(block->instance, args, "initialise_chunk_kernel_celly");
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
    int xdim0_initialise_chunk_kernel_celly = args[0].dat->size[0];
    int ydim0_initialise_chunk_kernel_celly = args[0].dat->size[1];
    int xdim1_initialise_chunk_kernel_celly = args[1].dat->size[0];
    int ydim1_initialise_chunk_kernel_celly = args[1].dat->size[1];
    int xdim2_initialise_chunk_kernel_celly = args[2].dat->size[0];
    int ydim2_initialise_chunk_kernel_celly = args[2].dat->size[1];

//  =======================================================
//  Set up initial pointers and exchange halos if necessary
//  =======================================================
    int base0 = args[0].dat->base_offset;
    double * __restrict__ vertexy_p = (double *)(args[0].data_d + base0);
    int base1 = args[1].dat->base_offset;
    double * __restrict__ celly_p = (double *)(args[1].data_d + base1);
    int base2 = args[2].dat->base_offset;
    double * __restrict__ celldy_p = (double *)(args[2].data_d + base2);

#ifndef OPS_LAZY
//  ==============
//  Halo Exchanges
//  ==============
    ops_H_D_exchanges_device(args, 3);
    ops_halo_exchanges(args,3,range);
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[50].mpi_time += __t2 - __t1;
    }

    int start_0 = start[0];
    int end_0 = end[0];
    int start_1 = start[1];
    int end_1 = end[1];
    int start_2 = start[2];
    int end_2 = end[2];
    if ((end[0]-start[0])>0 && (end[1]-start[1])>0 && (end[2]-start[2])>0) {
        block->instance->sycl_instance->queue->submit([&](cl::sycl::handler &cgh) {

          auto grid_sycl = (*grid_p).template get_access<cl::sycl::access::mode::read>(cgh);

          cgh.parallel_for<class initialise_chunk_kernel_celly_kernel>(cl::sycl::nd_range<3>(cl::sycl::range<3>(
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

              const ACC<double> vertexy(xdim0_initialise_chunk_kernel_celly, ydim0_initialise_chunk_kernel_celly, vertexy_p + n_x*0 + n_y * xdim0_initialise_chunk_kernel_celly*1 + n_z * xdim0_initialise_chunk_kernel_celly * ydim0_initialise_chunk_kernel_celly*0);
              ACC<double> celly(xdim1_initialise_chunk_kernel_celly, ydim1_initialise_chunk_kernel_celly, celly_p + n_x*0 + n_y * xdim1_initialise_chunk_kernel_celly*1 + n_z * xdim1_initialise_chunk_kernel_celly * ydim1_initialise_chunk_kernel_celly*0);
              ACC<double> celldy(xdim2_initialise_chunk_kernel_celly, ydim2_initialise_chunk_kernel_celly, celldy_p + n_x*0 + n_y * xdim2_initialise_chunk_kernel_celly*1 + n_z * xdim2_initialise_chunk_kernel_celly * ydim2_initialise_chunk_kernel_celly*0);
//  =========
//  User code
//  =========
                if (n_x < end_0 && n_y < end_1 && n_z < end_2) {
                    
  double d_y = (grid_sycl[0].ymax - grid_sycl[0].ymin)/(double)grid_sycl[0].y_cells;
  celly(0,0,0)  = 0.5*( vertexy(0,0,0) + vertexy(0,1,0) );
  celldy(0,0,0)  = d_y;
  if(celldy(0,0,0) < 0) {


  }

                }
            });
        });
    }

    if (block->instance->OPS_diags > 1) {
        block->instance->sycl_instance->queue->wait();
        ops_timers_core(&__c1, &__t1);
        block->instance->OPS_kernels[50].time += __t1 -__t2;
    }

#ifndef OPS_LAZY
    ops_set_dirtybit_device(args, 3);
    ops_set_halo_dirtybit3(&args[1], range);
    ops_set_halo_dirtybit3(&args[2], range);
#endif

    if (block->instance->OPS_diags > 1) {
//  ====================
//  Update kernel record
//  ====================
    ops_timers_core(&__c2, &__t2);
    block->instance->OPS_kernels[50].mpi_time += __t2 -__t1;
    block->instance->OPS_kernels[50].transfer += ops_compute_transfer(dim, start, end, &arg0);
    block->instance->OPS_kernels[50].transfer += ops_compute_transfer(dim, start, end, &arg1);
    block->instance->OPS_kernels[50].transfer += ops_compute_transfer(dim, start, end, &arg2);
}
}

#ifdef OPS_LAZY
void ops_par_loop_initialise_chunk_kernel_celly(
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

    create_kerneldesc_and_enque("initialise_chunk_kernel_celly", args, 3, 50, dim, 1, range, block, ops_par_loop_initialise_chunk_kernel_celly_execute);
}
#endif
