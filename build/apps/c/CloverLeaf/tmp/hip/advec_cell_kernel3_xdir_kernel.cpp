// Auto-generated at 2026-10-29 00:13:50.261708 by ops-translator legacy

__constant__ int dims_advec_cell_kernel3_xdir [8][1];
static int dims_advec_cell_kernel3_xdir_h [8][1] = {{0}};

//  =============
//  User function
//  =============
__device__ inline void advec_cell_kernel3_xdir_gpu(const ACC<double> &vol_flux_x,
  const ACC<double> &pre_vol,
  const ACC<int> &xx,
  const ACC<double> &vertexdx,
  const ACC<double> &density1,
  const ACC<double> &energy1,
  ACC<double> &mass_flux_x,
  ACC<double> &ener_flux) {

  double sigmat, sigmav, sigmam, sigma3, sigma4;
  double diffuw, diffdw, limiter;
  double one_by_six = 1.0/6.0;

  int x_max=field.x_max;

  int upwind,donor,downwind,dif;






  if(vol_flux_x(0,0) > 0.0) {
    upwind   = -2;
    donor    = -1;
    downwind = 0;
    dif      = donor;
  }
  else if (xx(1,0) < x_max+2-2) {
    upwind   = 1;
    donor    = 0;
    downwind = -1;
    dif      = upwind;
  } else {
    upwind   = 0;
    donor    = 0;
    downwind = -1;
    dif      = upwind;
  }


  sigmat = fabs(vol_flux_x(0,0))/pre_vol(donor,0);
  sigma3 = (1.0 + sigmat)*(vertexdx(0,0)/vertexdx(dif,0));
  sigma4 = 2.0 - sigmat;

  sigmav = sigmat;

  diffuw = density1(donor,0) - density1(upwind,0);
  diffdw = density1(downwind,0) - density1(donor,0);

  if( (diffuw*diffdw) > 0.0)
    limiter=(1.0 - sigmav) * SIGN(1.0 , diffdw) *
    MIN( MIN(fabs(diffuw), fabs(diffdw)),
    one_by_six * (sigma3*fabs(diffuw) + sigma4 * fabs(diffdw)));
  else
    limiter=0.0;

  mass_flux_x(0,0) = (vol_flux_x(0,0)) * ( density1(donor,0) + limiter );

  sigmam = fabs(mass_flux_x(0,0))/( density1(donor,0) * pre_vol(donor,0));
  diffuw = energy1(donor,0) - energy1(upwind,0);
  diffdw = energy1(downwind,0) - energy1(donor,0);

  if( (diffuw*diffdw) > 0.0)
    limiter = (1.0 - sigmam) * SIGN(1.0,diffdw) *
    MIN( MIN(fabs(diffuw), fabs(diffdw)),
    one_by_six * (sigma3 * fabs(diffuw) + sigma4 * fabs(diffdw)));
  else
    limiter=0.0;

  ener_flux(0,0) = mass_flux_x(0,0) * ( energy1(donor,0) + limiter );
}


//  ============================
//  Cuda kernel wrapper function
//  ============================
__global__ void ops_advec_cell_kernel3_xdir(
double* __restrict arg0,
double* __restrict arg1,
int* __restrict arg2,
double* __restrict arg3,
double* __restrict arg4,
double* __restrict arg5,
double* __restrict arg6,
double* __restrict arg7,
int size0,
int size1) {
    int idx_y = blockDim.y * blockIdx.y + threadIdx.y;
    int idx_x = blockDim.x * blockIdx.x + threadIdx.x;

    arg0 += idx_x * 1*1 + idx_y * 1*1 * dims_advec_cell_kernel3_xdir[0][0];
    arg1 += idx_x * 1*1 + idx_y * 1*1 * dims_advec_cell_kernel3_xdir[1][0];
    arg2 += idx_x * 1*1 + idx_y * 0*1 * dims_advec_cell_kernel3_xdir[2][0];
    arg3 += idx_x * 1*1 + idx_y * 0*1 * dims_advec_cell_kernel3_xdir[3][0];
    arg4 += idx_x * 1*1 + idx_y * 1*1 * dims_advec_cell_kernel3_xdir[4][0];
    arg5 += idx_x * 1*1 + idx_y * 1*1 * dims_advec_cell_kernel3_xdir[5][0];
    arg6 += idx_x * 1*1 + idx_y * 1*1 * dims_advec_cell_kernel3_xdir[6][0];
    arg7 += idx_x * 1*1 + idx_y * 1*1 * dims_advec_cell_kernel3_xdir[7][0];

    if (idx_x < size0 && idx_y < size1) {
        const ACC<double> argp0(dims_advec_cell_kernel3_xdir[0][0], arg0);
        const ACC<double> argp1(dims_advec_cell_kernel3_xdir[1][0], arg1);
        const ACC<int> argp2(dims_advec_cell_kernel3_xdir[2][0], arg2);
        const ACC<double> argp3(dims_advec_cell_kernel3_xdir[3][0], arg3);
        const ACC<double> argp4(dims_advec_cell_kernel3_xdir[4][0], arg4);
        const ACC<double> argp5(dims_advec_cell_kernel3_xdir[5][0], arg5);
        ACC<double> argp6(dims_advec_cell_kernel3_xdir[6][0], arg6);
        ACC<double> argp7(dims_advec_cell_kernel3_xdir[7][0], arg7);
        advec_cell_kernel3_xdir_gpu(
         argp0, argp1, argp2, argp3, argp4,
         argp5, argp6, argp7);
    }

} //End of cuda kernel wrapper function

//  ==================
//  Host stub function
//  ==================
#ifndef OPS_LAZY
void ops_par_loop_advec_cell_kernel3_xdir(char const *name, ops_block block, int dim, int* range,
 ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
 ops_arg arg4, ops_arg arg5, ops_arg arg6, ops_arg arg7) {
#else
void ops_par_loop_advec_cell_kernel3_xdir_execute(ops_kernel_descriptor *desc) {
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
    double __t1, __t2, __c1, __c2;

    ops_arg args[8];

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;
    args[3] = arg3;
    args[4] = arg4;
    args[5] = arg5;
    args[6] = arg6;
    args[7] = arg7;

#if CHECKPOINTING && !OPS_LAZY
    if (!ops_checkpointing_before(args, 8, range, 5)) return;
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timing_realloc(block->instance, 5, "advec_cell_kernel3_xdir");
        block->instance->OPS_kernels[5].count++;
        ops_timers_core(&__c1, &__t1);
    }

//  =================================================
//  compute locally allocated range for the sub-block
//  =================================================
    int start_indx[2];
    int end_indx[2];
#ifdef OPS_MPI
    int arg_idx[2];
#endif

#if defined(OPS_LAZY) || !defined(OPS_MPI)
    for (int n = 0; n < 2; n++) {
        start_indx[n] = range[2*n];
        end_indx[n]   = range[2*n+1];
    }
#else
    if (compute_ranges(args, 8, block, range, start_indx, end_indx, arg_idx) < 0) return;
#endif

    int xdim0 = args[0].dat->size[0];
    int xdim1 = args[1].dat->size[0];
    int xdim2 = args[2].dat->size[0];
    int xdim3 = args[3].dat->size[0];
    int xdim4 = args[4].dat->size[0];
    int xdim5 = args[5].dat->size[0];
    int xdim6 = args[6].dat->size[0];
    int xdim7 = args[7].dat->size[0];

    if (xdim0 != dims_advec_cell_kernel3_xdir_h[0][0] || xdim1 != dims_advec_cell_kernel3_xdir_h[1][0] || xdim2 != dims_advec_cell_kernel3_xdir_h[2][0] || xdim3 != dims_advec_cell_kernel3_xdir_h[3][0] || xdim4 != dims_advec_cell_kernel3_xdir_h[4][0] || xdim5 != dims_advec_cell_kernel3_xdir_h[5][0] || xdim6 != dims_advec_cell_kernel3_xdir_h[6][0] || xdim7 != dims_advec_cell_kernel3_xdir_h[7][0]) {
        dims_advec_cell_kernel3_xdir_h[0][0] = xdim0;
        dims_advec_cell_kernel3_xdir_h[1][0] = xdim1;
        dims_advec_cell_kernel3_xdir_h[2][0] = xdim2;
        dims_advec_cell_kernel3_xdir_h[3][0] = xdim3;
        dims_advec_cell_kernel3_xdir_h[4][0] = xdim4;
        dims_advec_cell_kernel3_xdir_h[5][0] = xdim5;
        dims_advec_cell_kernel3_xdir_h[6][0] = xdim6;
        dims_advec_cell_kernel3_xdir_h[7][0] = xdim7;

        hipSafeCall(block->instance->ostream(), hipMemcpyToSymbol( dims_advec_cell_kernel3_xdir, dims_advec_cell_kernel3_xdir_h, sizeof(dims_advec_cell_kernel3_xdir)));
    }


    int x_size = MAX(0,end_indx[0]-start_indx[0]);
    int y_size = MAX(0,end_indx[1]-start_indx[1]);

    dim3 grid( (x_size-1)/block->instance->OPS_block_size_x+ 1, (y_size-1)/block->instance->OPS_block_size_y + 1, 1);
    dim3 tblock(block->instance->OPS_block_size_x,block->instance->OPS_block_size_y,block->instance->OPS_block_size_z);



    long long int dat0 = (block->instance->OPS_soa ? args[0].dat->type_size : args[0].dat->elem_size);
    long long int dat1 = (block->instance->OPS_soa ? args[1].dat->type_size : args[1].dat->elem_size);
    long long int dat2 = (block->instance->OPS_soa ? args[2].dat->type_size : args[2].dat->elem_size);
    long long int dat3 = (block->instance->OPS_soa ? args[3].dat->type_size : args[3].dat->elem_size);
    long long int dat4 = (block->instance->OPS_soa ? args[4].dat->type_size : args[4].dat->elem_size);
    long long int dat5 = (block->instance->OPS_soa ? args[5].dat->type_size : args[5].dat->elem_size);
    long long int dat6 = (block->instance->OPS_soa ? args[6].dat->type_size : args[6].dat->elem_size);
    long long int dat7 = (block->instance->OPS_soa ? args[7].dat->type_size : args[7].dat->elem_size);

    char *p_a[8];

//  =======================
//  set up initial pointers
//  =======================
    long long int base0 = args[0].dat->base_offset + 
             dat0 * 1 * (start_indx[0] * args[0].stencil->stride[0]);
    base0 = base0 + dat0 *
                       args[0].dat->size[0] *
                      (start_indx[1] * args[0].stencil->stride[1]);
    p_a[0] = (char *)args[0].data_d + base0;

    long long int base1 = args[1].dat->base_offset + 
             dat1 * 1 * (start_indx[0] * args[1].stencil->stride[0]);
    base1 = base1 + dat1 *
                       args[1].dat->size[0] *
                      (start_indx[1] * args[1].stencil->stride[1]);
    p_a[1] = (char *)args[1].data_d + base1;

    long long int base2 = args[2].dat->base_offset + 
             dat2 * 1 * (start_indx[0] * args[2].stencil->stride[0]);
    base2 = base2 + dat2 *
                       args[2].dat->size[0] *
                      (start_indx[1] * args[2].stencil->stride[1]);
    p_a[2] = (char *)args[2].data_d + base2;

    long long int base3 = args[3].dat->base_offset + 
             dat3 * 1 * (start_indx[0] * args[3].stencil->stride[0]);
    base3 = base3 + dat3 *
                       args[3].dat->size[0] *
                      (start_indx[1] * args[3].stencil->stride[1]);
    p_a[3] = (char *)args[3].data_d + base3;

    long long int base4 = args[4].dat->base_offset + 
             dat4 * 1 * (start_indx[0] * args[4].stencil->stride[0]);
    base4 = base4 + dat4 *
                       args[4].dat->size[0] *
                      (start_indx[1] * args[4].stencil->stride[1]);
    p_a[4] = (char *)args[4].data_d + base4;

    long long int base5 = args[5].dat->base_offset + 
             dat5 * 1 * (start_indx[0] * args[5].stencil->stride[0]);
    base5 = base5 + dat5 *
                       args[5].dat->size[0] *
                      (start_indx[1] * args[5].stencil->stride[1]);
    p_a[5] = (char *)args[5].data_d + base5;

    long long int base6 = args[6].dat->base_offset + 
             dat6 * 1 * (start_indx[0] * args[6].stencil->stride[0]);
    base6 = base6 + dat6 *
                       args[6].dat->size[0] *
                      (start_indx[1] * args[6].stencil->stride[1]);
    p_a[6] = (char *)args[6].data_d + base6;

    long long int base7 = args[7].dat->base_offset + 
             dat7 * 1 * (start_indx[0] * args[7].stencil->stride[0]);
    base7 = base7 + dat7 *
                       args[7].dat->size[0] *
                      (start_indx[1] * args[7].stencil->stride[1]);
    p_a[7] = (char *)args[7].data_d + base7;

//  =============
//  Halo exchange
//  =============
#ifndef OPS_LAZY
    ops_H_D_exchanges_device(args, 8);
    ops_halo_exchanges(args, 8, range);
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[5].mpi_time += __t2 - __t1;
    }


//call kernel wrapper function, passing in pointers to data
    if (x_size > 0 && y_size > 0) {

        ops_advec_cell_kernel3_xdir<<<grid, tblock >>> ( 
         (double *)p_a[0], (double *)p_a[1],
         (int *)p_a[2], (double *)p_a[3],
         (double *)p_a[4], (double *)p_a[5],
         (double *)p_a[6], (double *)p_a[7],
        x_size, y_size);

    }

    hipSafeCall(block->instance->ostream(), hipGetLastError());

    if (block->instance->OPS_diags>1) {
        hipSafeCall(block->instance->ostream(), hipDeviceSynchronize());
        ops_timers_core(&__c1, &__t1);
        block->instance->OPS_kernels[5].time += __t1 - __t2;
    }

#ifndef OPS_LAZY
    ops_set_dirtybit_device(args, 8);
    ops_set_halo_dirtybit3(&args[6], range);
    ops_set_halo_dirtybit3(&args[7], range);
#endif

    if (block->instance->OPS_diags > 1) {
        //  ====================
        //  Update kernel record
        //  ====================
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[5].mpi_time += __t2 - __t1;
        block->instance->OPS_kernels[5].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg0);
        block->instance->OPS_kernels[5].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg1);
        block->instance->OPS_kernels[5].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg2);
        block->instance->OPS_kernels[5].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg3);
        block->instance->OPS_kernels[5].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg4);
        block->instance->OPS_kernels[5].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg5);
        block->instance->OPS_kernels[5].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg6);
        block->instance->OPS_kernels[5].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg7);
    }
}

#ifdef OPS_LAZY
void ops_par_loop_advec_cell_kernel3_xdir(char const *name, ops_block block, int dim, int* range,
 ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
 ops_arg arg4, ops_arg arg5, ops_arg arg6, ops_arg arg7)
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

    create_kerneldesc_and_enque("advec_cell_kernel3_xdir", args, 8, 5, dim, 1, range, block, ops_par_loop_advec_cell_kernel3_xdir_execute);
}
#endif
