// Auto-generated at 2026-10-29 00:14:13.126588 by ops-translator legacy


//  ==================
//  Host stub function
//  ==================
#ifndef OPS_LAZY
void ops_par_loop_Ker_Diffuse_Refl_Norm_xneg_ypos(
    const char *name,
    ops_block block,
    int dim,
    int* range,
    ops_arg arg0, 
    ops_arg arg1
)
{
#else
void ops_par_loop_Ker_Diffuse_Refl_Norm_xneg_ypos_execute(ops_kernel_descriptor *desc)
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
    double __t1, __t2, __c1, __c2;

    ops_arg args[2];

    args[0] = arg0;
    args[1] = arg1;

#if defined(CHECKPOINTING) && !defined(OPS_LAZY)
    if (!ops_checkpointing_before(args, 2, range, 7)) return;
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timing_realloc(block->instance, 7, "Ker_Diffuse_Refl_Norm_xneg_ypos");
        block->instance->OPS_kernels[7].count++;
        ops_timers_core(&__c1, &__t1);
    }

#ifdef OPS_DEBUG
    ops_register_args(block->instance, args, "Ker_Diffuse_Refl_Norm_xneg_ypos");
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

    int start0 = start[0];
    int end0 = end[0];
    int start1 = start[1];
    int end1 = end[1];

    //  =====================================================
    //  Initialize global variable with the dimension of dats
    //  =====================================================
    int xdim0_Ker_Diffuse_Refl_Norm_xneg_ypos = args[0].dat->size[0];
    int ydim0_Ker_Diffuse_Refl_Norm_xneg_ypos = args[0].dat->size[1];

//  =======================================================
//  Set up initial pointers and exchange halos if necessary
//  =======================================================
    int base0 = args[0].dat->base_offset;
    double * __restrict__ f_p = (double *)(args[0].data_d + base0);



    int consts_bytes = 0;

    double *arg1h = (double *)args[1].data;
    consts_bytes += ROUND_UP(args[1].dim*sizeof(double));

    reallocConstArrays(block->instance,consts_bytes);
    consts_bytes = 0;

    args[1].data = block->instance->OPS_consts_h + consts_bytes;
    args[1].data_d = block->instance->OPS_consts_d + consts_bytes;
    for (int d = 0; d < args[1].dim; d++) {
        ((double *)args[1].data)[d] = arg1h[d];
    }
    consts_bytes += ROUND_UP(args[1].dim*sizeof(double));

    mvConstArraysToDevice(block->instance,consts_bytes);

    double * __restrict__ u_w = (double *)args[1].data_d;

#ifndef OPS_LAZY
//  ==============
//  Halo Exchanges
//  ==============
    ops_H_D_exchanges_device(args, 2);
    ops_halo_exchanges(args, 2,range);
    ops_H_D_exchanges_device(args, 2);
#endif //OPS_LAZY

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[7].mpi_time += __t2 - __t1;
    }

    #pragma omp target teams distribute parallel for collapse(2)
    for (int n_y = start1; n_y < end1; n_y++) {
        for (int n_x = start0; n_x < end0; n_x++) {

#ifdef OPS_SOA
            ACC<double> f(9, xdim0_Ker_Diffuse_Refl_Norm_xneg_ypos, ydim0_Ker_Diffuse_Refl_Norm_xneg_ypos, f_p + n_x*1 + n_y * xdim0_Ker_Diffuse_Refl_Norm_xneg_ypos*1);
#else
            ACC<double> f(9, xdim0_Ker_Diffuse_Refl_Norm_xneg_ypos, ydim0_Ker_Diffuse_Refl_Norm_xneg_ypos, f_p + 9*(n_x*1 + n_y * xdim0_Ker_Diffuse_Refl_Norm_xneg_ypos*1));
#endif


  double u = u_w[0];
  double v = u_w[1];

  double sqrt3 = sqrt(3);

  double f2 = f(2, 0, 0);
  double f3 = f(3, 0, 0);
  double f6 = f(6, 0, 0);

  double rho_w = (-36 * (f2 + f3 + f6)) /
                  (-9. + 5. * sqrt3 * u - 3. * u * u -
                    5. * sqrt3 * v + 3. * u * v - 3. * v * v);
  f(8, 0, 0) = f6 + rho_w * (u - v) / (6. * sqrt3);
  f(1, 0, 0) = f3 + 2 * rho_w * u / (3. * sqrt3);
  f(4, 0, 0) = f2 - 2 * rho_w * v / (3. * sqrt3);
  f(0, 0, 0) = 2. * rho_w * (2. - u * u - v * v) / 9.;
  f(5, 0, 0) = (1. + u * u + sqrt3 * v + v * v +
                u * (sqrt3 + 3 * v)) * rho_w / 36.;
  f(7, 0, 0) = (1. - sqrt3 * u + u * u - sqrt3 * v + 3 * u * v + v * v) * rho_w / 36.;

}
}

if (block->instance->OPS_diags > 1) {
ops_timers_core(&__c1, &__t1);
block->instance->OPS_kernels[7].time += __t1 - __t2;
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
        block->instance->OPS_kernels[7].mpi_time += __t2 - __t1;
        block->instance->OPS_kernels[7].transfer += ops_compute_transfer(dim, start, end, &arg0);
    }
}

#ifdef OPS_LAZY
void ops_par_loop_Ker_Diffuse_Refl_Norm_xneg_ypos(
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

    create_kerneldesc_and_enque("Ker_Diffuse_Refl_Norm_xneg_ypos", args, 2, 7, dim, 1, range, block, ops_par_loop_Ker_Diffuse_Refl_Norm_xneg_ypos_execute);
}
#endif
