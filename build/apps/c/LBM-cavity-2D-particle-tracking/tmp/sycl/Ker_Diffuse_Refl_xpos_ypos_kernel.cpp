// Auto-generated at 2026-10-29 00:14:13.426676 by ops-translator legacy


//  ==================
//  Host stub function
//  ==================
#ifndef OPS_LAZY
void ops_par_loop_Ker_Diffuse_Refl_xpos_ypos(
    const char *name,
    ops_block block,
    int dim,
    int* range,
    ops_arg arg0, 
    ops_arg arg1
)
{
#else
void ops_par_loop_Ker_Diffuse_Refl_xpos_ypos_execute(ops_kernel_descriptor *desc)
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
    if (!ops_checkpointing_before(args,2, range, 6)) return;
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timing_realloc(block->instance, 6, "Ker_Diffuse_Refl_xpos_ypos");
        block->instance->OPS_kernels[6].count++;
        ops_timers_core(&__c1, &__t1);
    }

#ifdef OPS_DEBUG
    ops_register_args(block->instance, args, "Ker_Diffuse_Refl_xpos_ypos");
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
    if (compute_ranges(args, 2, block, range, start, end, arg_idx) < 0) return;
#endif


//  =====================================================
//  Initialize global variable with the dimension of dats
//  =====================================================
    int xdim0_Ker_Diffuse_Refl_xpos_ypos = args[0].dat->size[0];
    int ydim0_Ker_Diffuse_Refl_xpos_ypos = args[0].dat->size[1];

//  =======================================================
//  Set up initial pointers and exchange halos if necessary
//  =======================================================
    int base0 = args[0].dat->base_offset;
    double * __restrict__ f_p = (double *)(args[0].data_d + base0);
    double *arg1h = (double *)args[1].data;

    int consts_bytes = 0;

    consts_bytes += ROUND_UP(2*sizeof(double));
    reallocConstArrays(block->instance,consts_bytes);
    consts_bytes = 0;
    consts_bytes = 0;
    arg1.data = block->instance->OPS_consts_h + consts_bytes;
    double* arg1_data_d = (double*)( block->instance->OPS_consts_d + consts_bytes);
    for (int d=0; d<2; d++) ((double *)arg1.data)[d] = arg1h[d];
    consts_bytes += ROUND_UP(2*sizeof(double));
    mvConstArraysToDevice(block->instance,consts_bytes);

#ifndef OPS_LAZY
//  ==============
//  Halo Exchanges
//  ==============
    ops_H_D_exchanges_device(args, 2);
    ops_halo_exchanges(args,2,range);
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


          cgh.parallel_for<class Ker_Diffuse_Refl_xpos_ypos_kernel>(cl::sycl::nd_range<2>(cl::sycl::range<2>(
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

              #ifdef OPS_SOA
              ACC<double> f(9, xdim0_Ker_Diffuse_Refl_xpos_ypos, ydim0_Ker_Diffuse_Refl_xpos_ypos, f_p + n_x*1 + n_y * xdim0_Ker_Diffuse_Refl_xpos_ypos*1);
              #else
              ACC<double> f(9, xdim0_Ker_Diffuse_Refl_xpos_ypos, ydim0_Ker_Diffuse_Refl_xpos_ypos, f_p + 9*(n_x*1 + n_y * xdim0_Ker_Diffuse_Refl_xpos_ypos*1));
              #endif
              const double *u_w = arg1_data_d;
//  =========
//  User code
//  =========
                if (n_x < end_0 && n_y < end_1) {
                    

  double u = u_w[0];
  double v = u_w[1];

  double sqrt3 = cl::sycl::sqrt(3);

  double f1 = f(1, 0, 0);
  double f5 = f(5, 0, 0);
  double f2 = f(2, 0, 0);
  double rho_w = (36 * (f1 + f2 + f5)) /
                    (9 + 5 * sqrt3 * u + 3 * u * u +
                     5 * sqrt3 * v + 3 * u * v + 3 * v * v);

  f(3, 0, 0) = f1 - 2 * rho_w * u / (3 * sqrt3);
  f(4, 0, 0) = f2 - 2 * rho_w * v / (3 * sqrt3);
  f(7, 0, 0) = f5 - rho_w * (u + v) / (6 * sqrt3);
  f(0, 0, 0) = 2 * rho_w * (2 - u * u - v * v) / 9;
  f(6, 0, 0) =
      (1 + u * u + sqrt3 * v + v * v - u * (sqrt3 + 3 * v)) *
      rho_w / 36;
  f(8, 0, 0) =
      (1 + u * u + u * (sqrt3 - 3 * v) - sqrt3 * v + v * v) *
      rho_w / 36;

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
    ops_set_dirtybit_device(args, 2);
    ops_set_halo_dirtybit3(&args[0], range);
#endif

    if (block->instance->OPS_diags > 1) {
//  ====================
//  Update kernel record
//  ====================
    ops_timers_core(&__c2, &__t2);
    block->instance->OPS_kernels[6].mpi_time += __t2 -__t1;
    block->instance->OPS_kernels[6].transfer += ops_compute_transfer(dim, start, end, &arg0);
}
}

#ifdef OPS_LAZY
void ops_par_loop_Ker_Diffuse_Refl_xpos_ypos(
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

    create_kerneldesc_and_enque("Ker_Diffuse_Refl_xpos_ypos", args, 2, 6, dim, 1, range, block, ops_par_loop_Ker_Diffuse_Refl_xpos_ypos_execute);
}
#endif
