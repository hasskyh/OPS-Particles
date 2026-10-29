// Auto-generated at 2026-10-29 00:13:53.448248 by ops-translator legacy


//  ==================
//  Host stub function
//  ==================
#ifndef OPS_LAZY
void ops_par_loop_viscosity_kernel(
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
    ops_arg arg6
)
{
#else
void ops_par_loop_viscosity_kernel_execute(ops_kernel_descriptor *desc)
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
#endif

//  ======
//  Timing
//  ======
    double __t1,__t2,__c1,__c2;

    ops_arg args[7];

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;
    args[3] = arg3;
    args[4] = arg4;
    args[5] = arg5;
    args[6] = arg6;

#if defined(CHECKPOINTING) && !defined(OPS_LAZY)
    if (!ops_checkpointing_before(args,7, range, 82)) return;
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timing_realloc(block->instance, 82, "viscosity_kernel");
        block->instance->OPS_kernels[82].count++;
        ops_timers_core(&__c1, &__t1);
    }

#ifdef OPS_DEBUG
    ops_register_args(block->instance, args, "viscosity_kernel");
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
    if (compute_ranges(args, 7, block, range, start, end, arg_idx) < 0) return;
#endif


//  =====================================================
//  Initialize global variable with the dimension of dats
//  =====================================================
    int xdim0_viscosity_kernel = args[0].dat->size[0];
    int xdim1_viscosity_kernel = args[1].dat->size[0];
    int xdim2_viscosity_kernel = args[2].dat->size[0];
    int xdim3_viscosity_kernel = args[3].dat->size[0];
    int xdim4_viscosity_kernel = args[4].dat->size[0];
    int xdim5_viscosity_kernel = args[5].dat->size[0];
    int xdim6_viscosity_kernel = args[6].dat->size[0];

//  =======================================================
//  Set up initial pointers and exchange halos if necessary
//  =======================================================
    int base0 = args[0].dat->base_offset;
    double * __restrict__ xvel0_p = (double *)(args[0].data_d + base0);
    int base1 = args[1].dat->base_offset;
    double * __restrict__ yvel0_p = (double *)(args[1].data_d + base1);
    int base2 = args[2].dat->base_offset;
    double * __restrict__ celldx_p = (double *)(args[2].data_d + base2);
    int base3 = args[3].dat->base_offset;
    double * __restrict__ celldy_p = (double *)(args[3].data_d + base3);
    int base4 = args[4].dat->base_offset;
    double * __restrict__ pressure_p = (double *)(args[4].data_d + base4);
    int base5 = args[5].dat->base_offset;
    double * __restrict__ density0_p = (double *)(args[5].data_d + base5);
    int base6 = args[6].dat->base_offset;
    double * __restrict__ viscosity_p = (double *)(args[6].data_d + base6);

#ifndef OPS_LAZY
//  ==============
//  Halo Exchanges
//  ==============
    ops_H_D_exchanges_device(args, 7);
    ops_halo_exchanges(args,7,range);
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[82].mpi_time += __t2 - __t1;
    }

    int start_0 = start[0];
    int end_0 = end[0];
    int start_1 = start[1];
    int end_1 = end[1];
    if ((end[0]-start[0])>0 && (end[1]-start[1])>0) {
        block->instance->sycl_instance->queue->submit([&](cl::sycl::handler &cgh) {


          cgh.parallel_for<class viscosity_kernel_kernel>(cl::sycl::nd_range<2>(cl::sycl::range<2>(
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

              const ACC<double> xvel0(xdim0_viscosity_kernel, xvel0_p + n_x*1 + n_y * xdim0_viscosity_kernel*1);
              const ACC<double> yvel0(xdim1_viscosity_kernel, yvel0_p + n_x*1 + n_y * xdim1_viscosity_kernel*1);
              const ACC<double> celldx(xdim2_viscosity_kernel, celldx_p + n_x*1 + n_y * xdim2_viscosity_kernel*0);
              const ACC<double> celldy(xdim3_viscosity_kernel, celldy_p + n_x*0 + n_y * xdim3_viscosity_kernel*1);
              const ACC<double> pressure(xdim4_viscosity_kernel, pressure_p + n_x*1 + n_y * xdim4_viscosity_kernel*1);
              const ACC<double> density0(xdim5_viscosity_kernel, density0_p + n_x*1 + n_y * xdim5_viscosity_kernel*1);
              ACC<double> viscosity(xdim6_viscosity_kernel, viscosity_p + n_x*1 + n_y * xdim6_viscosity_kernel*1);
//  =========
//  User code
//  =========
                if (n_x < end_0 && n_y < end_1) {
                    

  double ugrad, vgrad,
         grad2,
         pgradx,pgrady,
         pgradx2,pgrady2,
         grad,
         ygrad, xgrad,
         div,
         strain2,
         limiter,
         pgrad;



  ugrad = (xvel0(1,0) + xvel0(1,1)) - (xvel0(0,0) + xvel0(0,1));
  vgrad = (yvel0(0,1) + yvel0(1,1)) - (yvel0(0,0) + yvel0(1,0));

  div = (celldx(0,0))*(ugrad) + (celldy(0,0))*(vgrad);

  strain2 = 0.5*(xvel0(0,1) + xvel0(1,1) - xvel0(0,0) - xvel0(1,0))/(celldy(0,0)) +
            0.5*(yvel0(1,0) + yvel0(1,1) - yvel0(0,0) - yvel0(0,1))/(celldx(0,0));



  pgradx  = (pressure(1,0) - pressure(-1,0))/(celldx(0,0)+ celldx(1,0));
  pgrady = (pressure(0,1) - pressure(0,-1))/(celldy(0,0)+ celldy(0,1));

  pgradx2 = pgradx * pgradx;
  pgrady2 = pgrady * pgrady;

  limiter = ((0.5*(ugrad)/celldx(0,0)) * pgradx2 +
             (0.5*(vgrad)/celldy(0,0)) * pgrady2 +
              strain2 * pgradx * pgrady)/ MAX(pgradx2 + pgrady2 , 1.0e-16);

  if( (limiter > 0.0) || (div >= 0.0)) {
        viscosity(0,0) = 0.0;
  }
  else {
    pgradx = SIGN( MAX(1.0e-16, cl::sycl::fabs(pgradx)), pgradx);
    pgrady = SIGN( MAX(1.0e-16, cl::sycl::fabs(pgrady)), pgrady);
    pgrad = cl::sycl::sqrt(pgradx*pgradx + pgrady*pgrady);
    xgrad = cl::sycl::fabs(celldx(0,0) * pgrad/pgradx);
    ygrad = cl::sycl::fabs(celldy(0,0) * pgrad/pgrady);
    grad  = MIN(xgrad,ygrad);
    grad2 = grad*grad;

    viscosity(0,0) = 2.0 * (density0(0,0)) * grad2 * limiter * limiter;
  }

                }
            });
        });
    }

    if (block->instance->OPS_diags > 1) {
        block->instance->sycl_instance->queue->wait();
        ops_timers_core(&__c1, &__t1);
        block->instance->OPS_kernels[82].time += __t1 -__t2;
    }

#ifndef OPS_LAZY
    ops_set_dirtybit_device(args, 7);
    ops_set_halo_dirtybit3(&args[6], range);
#endif

    if (block->instance->OPS_diags > 1) {
//  ====================
//  Update kernel record
//  ====================
    ops_timers_core(&__c2, &__t2);
    block->instance->OPS_kernels[82].mpi_time += __t2 -__t1;
    block->instance->OPS_kernels[82].transfer += ops_compute_transfer(dim, start, end, &arg0);
    block->instance->OPS_kernels[82].transfer += ops_compute_transfer(dim, start, end, &arg1);
    block->instance->OPS_kernels[82].transfer += ops_compute_transfer(dim, start, end, &arg2);
    block->instance->OPS_kernels[82].transfer += ops_compute_transfer(dim, start, end, &arg3);
    block->instance->OPS_kernels[82].transfer += ops_compute_transfer(dim, start, end, &arg4);
    block->instance->OPS_kernels[82].transfer += ops_compute_transfer(dim, start, end, &arg5);
    block->instance->OPS_kernels[82].transfer += ops_compute_transfer(dim, start, end, &arg6);
}
}

#ifdef OPS_LAZY
void ops_par_loop_viscosity_kernel(
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
    ops_arg arg6
  )
{
    ops_arg args[7];

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;
    args[3] = arg3;
    args[4] = arg4;
    args[5] = arg5;
    args[6] = arg6;

    create_kerneldesc_and_enque("viscosity_kernel", args, 7, 82, dim, 1, range, block, ops_par_loop_viscosity_kernel_execute);
}
#endif
