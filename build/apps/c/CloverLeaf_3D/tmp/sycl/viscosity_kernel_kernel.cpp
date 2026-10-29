// Auto-generated at 2026-10-29 00:14:07.631157 by ops-translator legacy


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
    ops_arg arg6, 
    ops_arg arg7, 
    ops_arg arg8, 
    ops_arg arg9, 
    ops_arg arg10, 
    ops_arg arg11
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
    ops_arg arg7 = desc->args[7];
    ops_arg arg8 = desc->args[8];
    ops_arg arg9 = desc->args[9];
    ops_arg arg10 = desc->args[10];
    ops_arg arg11 = desc->args[11];
#endif

//  ======
//  Timing
//  ======
    double __t1,__t2,__c1,__c2;

    ops_arg args[12];

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

#if defined(CHECKPOINTING) && !defined(OPS_LAZY)
    if (!ops_checkpointing_before(args,12, range, 140)) return;
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timing_realloc(block->instance, 140, "viscosity_kernel");
        block->instance->OPS_kernels[140].count++;
        ops_timers_core(&__c1, &__t1);
    }

#ifdef OPS_DEBUG
    ops_register_args(block->instance, args, "viscosity_kernel");
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
    if (compute_ranges(args, 12, block, range, start, end, arg_idx) < 0) return;
#endif


//  =====================================================
//  Initialize global variable with the dimension of dats
//  =====================================================
    int xdim0_viscosity_kernel = args[0].dat->size[0];
    int ydim0_viscosity_kernel = args[0].dat->size[1];
    int xdim1_viscosity_kernel = args[1].dat->size[0];
    int ydim1_viscosity_kernel = args[1].dat->size[1];
    int xdim2_viscosity_kernel = args[2].dat->size[0];
    int ydim2_viscosity_kernel = args[2].dat->size[1];
    int xdim3_viscosity_kernel = args[3].dat->size[0];
    int ydim3_viscosity_kernel = args[3].dat->size[1];
    int xdim4_viscosity_kernel = args[4].dat->size[0];
    int ydim4_viscosity_kernel = args[4].dat->size[1];
    int xdim5_viscosity_kernel = args[5].dat->size[0];
    int ydim5_viscosity_kernel = args[5].dat->size[1];
    int xdim6_viscosity_kernel = args[6].dat->size[0];
    int ydim6_viscosity_kernel = args[6].dat->size[1];
    int xdim7_viscosity_kernel = args[7].dat->size[0];
    int ydim7_viscosity_kernel = args[7].dat->size[1];
    int xdim8_viscosity_kernel = args[8].dat->size[0];
    int ydim8_viscosity_kernel = args[8].dat->size[1];
    int xdim9_viscosity_kernel = args[9].dat->size[0];
    int ydim9_viscosity_kernel = args[9].dat->size[1];
    int xdim10_viscosity_kernel = args[10].dat->size[0];
    int ydim10_viscosity_kernel = args[10].dat->size[1];
    int xdim11_viscosity_kernel = args[11].dat->size[0];
    int ydim11_viscosity_kernel = args[11].dat->size[1];

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
    int base7 = args[7].dat->base_offset;
    double * __restrict__ zvel0_p = (double *)(args[7].data_d + base7);
    int base8 = args[8].dat->base_offset;
    double * __restrict__ celldz_p = (double *)(args[8].data_d + base8);
    int base9 = args[9].dat->base_offset;
    double * __restrict__ xarea_p = (double *)(args[9].data_d + base9);
    int base10 = args[10].dat->base_offset;
    double * __restrict__ yarea_p = (double *)(args[10].data_d + base10);
    int base11 = args[11].dat->base_offset;
    double * __restrict__ zarea_p = (double *)(args[11].data_d + base11);

#ifndef OPS_LAZY
//  ==============
//  Halo Exchanges
//  ==============
    ops_H_D_exchanges_device(args, 12);
    ops_halo_exchanges(args,12,range);
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[140].mpi_time += __t2 - __t1;
    }

    int start_0 = start[0];
    int end_0 = end[0];
    int start_1 = start[1];
    int end_1 = end[1];
    int start_2 = start[2];
    int end_2 = end[2];
    if ((end[0]-start[0])>0 && (end[1]-start[1])>0 && (end[2]-start[2])>0) {
        block->instance->sycl_instance->queue->submit([&](cl::sycl::handler &cgh) {


          cgh.parallel_for<class viscosity_kernel_kernel>(cl::sycl::nd_range<3>(cl::sycl::range<3>(
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

              const ACC<double> xvel0(xdim0_viscosity_kernel, ydim0_viscosity_kernel, xvel0_p + n_x*1 + n_y * xdim0_viscosity_kernel*1 + n_z * xdim0_viscosity_kernel * ydim0_viscosity_kernel*1);
              const ACC<double> yvel0(xdim1_viscosity_kernel, ydim1_viscosity_kernel, yvel0_p + n_x*1 + n_y * xdim1_viscosity_kernel*1 + n_z * xdim1_viscosity_kernel * ydim1_viscosity_kernel*1);
              const ACC<double> celldx(xdim2_viscosity_kernel, ydim2_viscosity_kernel, celldx_p + n_x*1 + n_y * xdim2_viscosity_kernel*0 + n_z * xdim2_viscosity_kernel * ydim2_viscosity_kernel*0);
              const ACC<double> celldy(xdim3_viscosity_kernel, ydim3_viscosity_kernel, celldy_p + n_x*0 + n_y * xdim3_viscosity_kernel*1 + n_z * xdim3_viscosity_kernel * ydim3_viscosity_kernel*0);
              const ACC<double> pressure(xdim4_viscosity_kernel, ydim4_viscosity_kernel, pressure_p + n_x*1 + n_y * xdim4_viscosity_kernel*1 + n_z * xdim4_viscosity_kernel * ydim4_viscosity_kernel*1);
              const ACC<double> density0(xdim5_viscosity_kernel, ydim5_viscosity_kernel, density0_p + n_x*1 + n_y * xdim5_viscosity_kernel*1 + n_z * xdim5_viscosity_kernel * ydim5_viscosity_kernel*1);
              ACC<double> viscosity(xdim6_viscosity_kernel, ydim6_viscosity_kernel, viscosity_p + n_x*1 + n_y * xdim6_viscosity_kernel*1 + n_z * xdim6_viscosity_kernel * ydim6_viscosity_kernel*1);
              const ACC<double> zvel0(xdim7_viscosity_kernel, ydim7_viscosity_kernel, zvel0_p + n_x*1 + n_y * xdim7_viscosity_kernel*1 + n_z * xdim7_viscosity_kernel * ydim7_viscosity_kernel*1);
              const ACC<double> celldz(xdim8_viscosity_kernel, ydim8_viscosity_kernel, celldz_p + n_x*0 + n_y * xdim8_viscosity_kernel*0 + n_z * xdim8_viscosity_kernel * ydim8_viscosity_kernel*1);
              const ACC<double> xarea(xdim9_viscosity_kernel, ydim9_viscosity_kernel, xarea_p + n_x*1 + n_y * xdim9_viscosity_kernel*1 + n_z * xdim9_viscosity_kernel * ydim9_viscosity_kernel*1);
              const ACC<double> yarea(xdim10_viscosity_kernel, ydim10_viscosity_kernel, yarea_p + n_x*1 + n_y * xdim10_viscosity_kernel*1 + n_z * xdim10_viscosity_kernel * ydim10_viscosity_kernel*1);
              const ACC<double> zarea(xdim11_viscosity_kernel, ydim11_viscosity_kernel, zarea_p + n_x*1 + n_y * xdim11_viscosity_kernel*1 + n_z * xdim11_viscosity_kernel * ydim11_viscosity_kernel*1);
//  =========
//  User code
//  =========
                if (n_x < end_0 && n_y < end_1 && n_z < end_2) {
                    

  double grad2,
         pgradx,pgrady,pgradz,
         pgradx2,pgrady2,pgradz2,
         grad,
         ygrad, xgrad, zgrad,
         div,
         limiter,
         pgrad;


  double ugradx1=xvel0(0,0,0)+xvel0(0,1,0)+xvel0(0,0,1)+xvel0(0,1,1);
  double ugradx2=xvel0(1,0,0)+xvel0(1,1,0)+xvel0(1,0,1)+xvel0(1,1,1);
  double ugrady1=xvel0(0,0,0)+xvel0(1,0,0)+xvel0(0,0,1)+xvel0(1,0,1);
  double ugrady2=xvel0(0,1,0)+xvel0(1,1,0)+xvel0(0,1,1)+xvel0(1,1,1);
  double ugradz1=xvel0(0,0,0)+xvel0(1,0,0)+xvel0(0,1,0)+xvel0(1,1,0);
  double ugradz2=xvel0(0,0,1)+xvel0(1,0,1)+xvel0(0,1,1)+xvel0(1,1,1);

  double vgradx1=yvel0(0,0,0)+yvel0(0,1,0)+yvel0(0,0,1)+yvel0(0,1,1);
  double vgradx2=yvel0(1,0,0)+yvel0(1,1,0)+yvel0(1,0,1)+yvel0(1,1,1);
  double vgrady1=yvel0(0,0,0)+yvel0(1,0,0)+yvel0(0,0,1)+yvel0(1,0,1);
  double vgrady2=yvel0(0,1,0)+yvel0(1,1,0)+yvel0(0,1,1)+yvel0(1,1,1);
  double vgradz1=yvel0(0,0,0)+yvel0(1,0,0)+yvel0(0,1,0)+yvel0(1,1,0);
  double vgradz2=yvel0(0,0,1)+yvel0(1,0,1)+yvel0(0,1,1)+yvel0(1,1,1);

  double wgradx1=zvel0(0,0,0)+zvel0(0,1,0)+zvel0(0,0,1)+zvel0(0,1,1);
  double wgradx2=zvel0(1,0,0)+zvel0(1,1,0)+zvel0(1,0,1)+zvel0(1,1,1);
  double wgrady1=zvel0(0,0,0)+zvel0(1,0,0)+zvel0(0,0,1)+zvel0(1,0,1);
  double wgrady2=zvel0(0,1,0)+zvel0(1,1,0)+zvel0(0,1,1)+zvel0(1,1,1);
  double wgradz1=zvel0(0,0,0)+zvel0(1,0,0)+zvel0(0,1,0)+zvel0(1,1,0);
  double wgradz2=zvel0(0,0,1)+zvel0(1,0,1)+zvel0(0,1,1)+zvel0(1,1,1);

  div = xarea(0,0,0)*(ugradx2-ugradx1) + yarea(0,0,0)*(vgrady2-vgrady1) + zarea(0,0,0)*(wgradz2-wgradz1);

  double xx = 0.25*(ugradx2-ugradx1)/(celldx(0,0,0));
  double yy = 0.25*(vgrady2-vgrady1)/(celldy(0,0,0));
  double zz = 0.25*(wgradz2-wgradz1)/(celldz(0,0,0));
  double xy = 0.25*(ugrady2-ugrady1)/(celldy(0,0,0))+0.25*(vgradx2-vgradx1)/(celldx(0,0,0));
  double xz = 0.25*(ugradz2-ugradz1)/(celldz(0,0,0))+0.25*(wgradx2-wgradx1)/(celldx(0,0,0));
  double yz = 0.25*(vgradz2-vgradz1)/(celldz(0,0,0))+0.25*(wgrady2-wgrady1)/(celldy(0,0,0));



  pgradx = (pressure(1,0,0) - pressure(-1,0,0))/(celldx(0,0,0)+ celldx(1,0,0));
  pgrady = (pressure(0,1,0) - pressure(0,-1,0))/(celldy(0,0,0)+ celldy(0,1,0));
  pgradz = (pressure(0,0,1) - pressure(0,0,-1))/(celldz(0,0,0)+ celldz(0,0,1));

  pgradx2 = pgradx * pgradx;
  pgrady2 = pgrady * pgrady;
  pgradz2 = pgradz * pgradz;
  limiter = (xx*pgradx2+yy*pgrady2+zz*pgradz2
          +  xy*pgradx*pgrady+xz*pgradx*pgradz+yz*pgrady*pgradz)
                / MAX(pgradx2+pgrady2+pgradz2,1.0e-16);

  if( (limiter > 0.0) || (div >= 0.0)) {
        viscosity(0,0,0) = 0.0;
  }
  else {
    pgradx = SIGN( MAX(1.0e-16, cl::sycl::fabs(pgradx)), pgradx);
    pgrady = SIGN( MAX(1.0e-16, cl::sycl::fabs(pgrady)), pgrady);
    pgradz = SIGN( MAX(1.0e-16, cl::sycl::fabs(pgradz)), pgradz);
    pgrad = cl::sycl::sqrt(pgradx*pgradx + pgrady*pgrady + pgradz*pgradz);
    xgrad = cl::sycl::fabs(celldx(0,0,0) * pgrad/pgradx);
    ygrad = cl::sycl::fabs(celldy(0,0,0) * pgrad/pgrady);
    zgrad = cl::sycl::fabs(celldz(0,0,0) * pgrad/pgradz);
    grad  = MIN(xgrad,MIN(ygrad,zgrad));
    grad2 = grad*grad;

    viscosity(0,0,0) = 2.0 * (density0(0,0,0)) * grad2 * limiter * limiter;
  }

                }
            });
        });
    }

    if (block->instance->OPS_diags > 1) {
        block->instance->sycl_instance->queue->wait();
        ops_timers_core(&__c1, &__t1);
        block->instance->OPS_kernels[140].time += __t1 -__t2;
    }

#ifndef OPS_LAZY
    ops_set_dirtybit_device(args, 12);
    ops_set_halo_dirtybit3(&args[6], range);
#endif

    if (block->instance->OPS_diags > 1) {
//  ====================
//  Update kernel record
//  ====================
    ops_timers_core(&__c2, &__t2);
    block->instance->OPS_kernels[140].mpi_time += __t2 -__t1;
    block->instance->OPS_kernels[140].transfer += ops_compute_transfer(dim, start, end, &arg0);
    block->instance->OPS_kernels[140].transfer += ops_compute_transfer(dim, start, end, &arg1);
    block->instance->OPS_kernels[140].transfer += ops_compute_transfer(dim, start, end, &arg2);
    block->instance->OPS_kernels[140].transfer += ops_compute_transfer(dim, start, end, &arg3);
    block->instance->OPS_kernels[140].transfer += ops_compute_transfer(dim, start, end, &arg4);
    block->instance->OPS_kernels[140].transfer += ops_compute_transfer(dim, start, end, &arg5);
    block->instance->OPS_kernels[140].transfer += ops_compute_transfer(dim, start, end, &arg6);
    block->instance->OPS_kernels[140].transfer += ops_compute_transfer(dim, start, end, &arg7);
    block->instance->OPS_kernels[140].transfer += ops_compute_transfer(dim, start, end, &arg8);
    block->instance->OPS_kernels[140].transfer += ops_compute_transfer(dim, start, end, &arg9);
    block->instance->OPS_kernels[140].transfer += ops_compute_transfer(dim, start, end, &arg10);
    block->instance->OPS_kernels[140].transfer += ops_compute_transfer(dim, start, end, &arg11);
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
    ops_arg arg6,
    ops_arg arg7,
    ops_arg arg8,
    ops_arg arg9,
    ops_arg arg10,
    ops_arg arg11
  )
{
    ops_arg args[12];

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

    create_kerneldesc_and_enque("viscosity_kernel", args, 12, 140, dim, 1, range, block, ops_par_loop_viscosity_kernel_execute);
}
#endif
