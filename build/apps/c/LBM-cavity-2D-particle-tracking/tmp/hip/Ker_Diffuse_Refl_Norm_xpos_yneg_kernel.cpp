// Auto-generated at 2026-10-29 00:14:13.244223 by ops-translator legacy

__constant__ int dims_Ker_Diffuse_Refl_Norm_xpos_yneg [2][1];
static int dims_Ker_Diffuse_Refl_Norm_xpos_yneg_h [2][1] = {{0}};

//  =============
//  User function
//  =============
__device__ void Ker_Diffuse_Refl_Norm_xpos_yneg_gpu(ACC<double>& f,
  const double *u_w) {

  double u = u_w[0];
  double v = u_w[1];

  double sqrt3 = sqrt(3);

  double f1 = f(1, 0, 0);
  double f4 = f(4, 0, 0);
  double f8 = f(8, 0, 0);
  double rho_w = (36 * (f1 + f4 + f8)) /
                    (9. + 5. * sqrt3 * u + 3. * u * u -
                     5. * sqrt3 * v - 3. * u * v + 3. * v * v);

  f(6, 0, 0) = f8 + rho_w * (v - u) / (6 * sqrt3);
  f(3, 0, 0) = f1 - 2 * rho_w * u / (3 * sqrt3);
  f(2, 0, 0) = f4 + 2 * rho_w * v / (3 * sqrt3);
  f(0, 0, 0) = 2 * rho_w * (2 - u * u - v * v) / 9;
  f(5, 0, 0) = (1 + u * u + sqrt3 * v + v * v +
                             u * (sqrt3 + 3 * v))* rho_w /
                            36;
  f(7, 0, 0) =
      (1 - sqrt3 * u + u * u - sqrt3 * v + 3 * u * v + v * v) *
      rho_w / 36;
}


//  ============================
//  Cuda kernel wrapper function
//  ============================
__global__ void ops_Ker_Diffuse_Refl_Norm_xpos_yneg(
double* __restrict arg0,
const double* __restrict arg1,
int size0,
int size1) {
    int idx_y = blockDim.y * blockIdx.y + threadIdx.y;
    int idx_x = blockDim.x * blockIdx.x + threadIdx.x;

    arg0 += idx_x * 1*9 + idx_y * 1*9 * dims_Ker_Diffuse_Refl_Norm_xpos_yneg[0][0];

    if (idx_x < size0 && idx_y < size1) {
        ACC<double> argp0(9, dims_Ker_Diffuse_Refl_Norm_xpos_yneg[0][0], 0, arg0);
        Ker_Diffuse_Refl_Norm_xpos_yneg_gpu(
         argp0, arg1);
    }

} //End of cuda kernel wrapper function

//  ==================
//  Host stub function
//  ==================
#ifndef OPS_LAZY
void ops_par_loop_Ker_Diffuse_Refl_Norm_xpos_yneg(char const *name, ops_block block, int dim, int* range,
 ops_arg arg0, ops_arg arg1) {
#else
void ops_par_loop_Ker_Diffuse_Refl_Norm_xpos_yneg_execute(ops_kernel_descriptor *desc) {
    ops_block block = desc->block;
    int dim = desc->dim;
    int *range = desc->range;
    ops_arg arg0 = desc->args[0];
    ops_arg arg1 = desc->args[1];
#endif

//  ======
//  Timing
//  ======
    double __t1, __t2, __c1, __c2;

    ops_arg args[2];

    args[0] = arg0;
    args[1] = arg1;

#if CHECKPOINTING && !OPS_LAZY
    if (!ops_checkpointing_before(args, 2, range, 5)) return;
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timing_realloc(block->instance, 5, "Ker_Diffuse_Refl_Norm_xpos_yneg");
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
    if (compute_ranges(args, 2, block, range, start_indx, end_indx, arg_idx) < 0) return;
#endif

    int xdim0 = args[0].dat->size[0];

    if (xdim0 != dims_Ker_Diffuse_Refl_Norm_xpos_yneg_h[0][0]) {
        dims_Ker_Diffuse_Refl_Norm_xpos_yneg_h[0][0] = xdim0;

        hipSafeCall(block->instance->ostream(), hipMemcpyToSymbol( dims_Ker_Diffuse_Refl_Norm_xpos_yneg, dims_Ker_Diffuse_Refl_Norm_xpos_yneg_h, sizeof(dims_Ker_Diffuse_Refl_Norm_xpos_yneg)));
    }

    double *arg1h = (double *)arg1.data;

    int x_size = MAX(0,end_indx[0]-start_indx[0]);
    int y_size = MAX(0,end_indx[1]-start_indx[1]);

    dim3 grid( (x_size-1)/block->instance->OPS_block_size_x+ 1, (y_size-1)/block->instance->OPS_block_size_y + 1, 1);
    dim3 tblock(block->instance->OPS_block_size_x,block->instance->OPS_block_size_y,block->instance->OPS_block_size_z);

    int consts_bytes = 0;

    consts_bytes += ROUND_UP(arg1.dim*sizeof(double));

    reallocConstArrays(block->instance,consts_bytes);
    consts_bytes = 0;
    arg1.data = block->instance->OPS_consts_h + consts_bytes;
    arg1.data_d = block->instance->OPS_consts_d + consts_bytes;
    for (int d=0; d<arg1.dim; d++) ((double *)arg1.data)[d] = arg1h[d];
    consts_bytes += ROUND_UP(arg1.dim*sizeof(double));
    mvConstArraysToDevice(block->instance,consts_bytes);

    long long int dat0 = (block->instance->OPS_soa ? args[0].dat->type_size : args[0].dat->elem_size);

    char *p_a[2];

//  =======================
//  set up initial pointers
//  =======================
    long long int base0 = args[0].dat->base_offset + 
             dat0 * 1 * (start_indx[0] * args[0].stencil->stride[0]);
    base0 = base0 + dat0 *
                       args[0].dat->size[0] *
                      (start_indx[1] * args[0].stencil->stride[1]);
    p_a[0] = (char *)args[0].data_d + base0;

//  =============
//  Halo exchange
//  =============
#ifndef OPS_LAZY
    ops_H_D_exchanges_device(args, 2);
    ops_halo_exchanges(args, 2, range);
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[5].mpi_time += __t2 - __t1;
    }


//call kernel wrapper function, passing in pointers to data
    if (x_size > 0 && y_size > 0) {

        ops_Ker_Diffuse_Refl_Norm_xpos_yneg<<<grid, tblock >>> ( 
         (double *)p_a[0], (double *)arg1.data_d,
        x_size, y_size);

    }

    hipSafeCall(block->instance->ostream(), hipGetLastError());

    if (block->instance->OPS_diags>1) {
        hipSafeCall(block->instance->ostream(), hipDeviceSynchronize());
        ops_timers_core(&__c1, &__t1);
        block->instance->OPS_kernels[5].time += __t1 - __t2;
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
        block->instance->OPS_kernels[5].mpi_time += __t2 - __t1;
        block->instance->OPS_kernels[5].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg0);
    }
}

#ifdef OPS_LAZY
void ops_par_loop_Ker_Diffuse_Refl_Norm_xpos_yneg(char const *name, ops_block block, int dim, int* range,
 ops_arg arg0, ops_arg arg1)
{
    ops_arg args[2];

    args[0] = arg0;
    args[1] = arg1;

    create_kerneldesc_and_enque("Ker_Diffuse_Refl_Norm_xpos_yneg", args, 2, 5, dim, 1, range, block, ops_par_loop_Ker_Diffuse_Refl_Norm_xpos_yneg_execute);
}
#endif
