// Auto-generated at 2026-10-29 00:14:11.312252 by ops-translator legacy


//  ==================
//  Host stub function
//  ==================
#ifndef OPS_LAZY
void ops_par_loop_init_kernel(
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
    ops_arg arg7
)
{
#else
void ops_par_loop_init_kernel_execute(ops_kernel_descriptor *desc)
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

#if defined(CHECKPOINTING) && !defined(OPS_LAZY)
    if (!ops_checkpointing_before(args, 8, range, 1)) return;
#endif

    if (block->instance->OPS_diags > 1) {
        ops_timing_realloc(block->instance, 1, "init_kernel");
        block->instance->OPS_kernels[1].count++;
        ops_timers_core(&__c1, &__t1);
    }

#ifdef OPS_DEBUG
    ops_register_args(block->instance, args, "init_kernel");
#endif


//  =================================================
//  compute locally allocated range for the sub-block
//  =================================================
    int start[1];
    int end[1];
#if defined(OPS_MPI) && !defined(OPS_LAZY)
    int arg_idx[1];
#endif

#if defined(OPS_LAZY) || !defined(OPS_MPI)
    for (int n = 0; n < 1; n++) {
        start[n] = range[2*n];
        end[n]   = range[2*n+1];
    }
#else
    if (compute_ranges(args, 8, block, range, start, end, arg_idx) < 0) return;
#endif

    int start0 = start[0];
    int end0 = end[0];

    //  =====================================================
    //  Initialize global variable with the dimension of dats
    //  =====================================================

//  =======================================================
//  Set up initial pointers and exchange halos if necessary
//  =======================================================
    int base0 = args[0].dat->base_offset;
    double * __restrict__ x_p = (double *)(args[0].data_d + base0);

    int base1 = args[1].dat->base_offset;
    double * __restrict__ rho_new_p = (double *)(args[1].data_d + base1);

    int base2 = args[2].dat->base_offset;
    double * __restrict__ rhou_new_p = (double *)(args[2].data_d + base2);

    int base3 = args[3].dat->base_offset;
    double * __restrict__ rhoE_new_p = (double *)(args[3].data_d + base3);

    int base4 = args[4].dat->base_offset;
    double * __restrict__ rhoin_p = (double *)(args[4].data_d + base4);

    int base5 = args[5].dat->base_offset;
    double * __restrict__ rho_old_p = (double *)(args[5].data_d + base5);

    int base6 = args[6].dat->base_offset;
    double * __restrict__ rhou_old_p = (double *)(args[6].data_d + base6);

    int base7 = args[7].dat->base_offset;
    double * __restrict__ rhoE_old_p = (double *)(args[7].data_d + base7);

#ifndef OPS_LAZY
//  ==============
//  Halo Exchanges
//  ==============
    ops_H_D_exchanges_device(args, 8);
    ops_halo_exchanges(args, 8,range);
    ops_H_D_exchanges_device(args, 8);
#endif //OPS_LAZY

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[1].mpi_time += __t2 - __t1;
    }

    #pragma omp target teams distribute parallel for collapse(1)
    for (int n_x = start0; n_x < end0; n_x++) {

        const ACC<double> x(x_p + n_x*1);

        ACC<double> rho_new(rho_new_p + n_x*1);

        ACC<double> rhou_new(rhou_new_p + n_x*1);

        ACC<double> rhoE_new(rhoE_new_p + n_x*1);

        ACC<double> rhoin(rhoin_p + n_x*1);

        ACC<double> rho_old(rho_old_p + n_x*1);

        ACC<double> rhou_old(rhou_old_p + n_x*1);

        ACC<double> rhoE_old(rhoE_old_p + n_x*1);

        
  if (x(0) >= -4.0){
    rho_new(0) = 1.0 + eps * sin(lambda *x(0));
    rhou_new(0) = ur * rho_new(0);
    rhoE_new(0) = (pr / gam1) + 0.5 * pow(rhou_new(0),2)/rho_new(0);
  }
  else {
    rho_new(0) = rhol;
    rhou_new(0) = ul2 * rho_new(0);
    rhoE_new(0) = (pl / gam1) + 0.5 * pow(rhou_new(0),2)/rho_new(0);
  }
  rho_old(0)  = rho_new(0);
  rhou_old(0) = rhou_new(0);
  rhoE_old(0) = rhoE_new(0);

  rhoin(0) = rho_new(0);


    }

    if (block->instance->OPS_diags > 1) {
        ops_timers_core(&__c1, &__t1);
        block->instance->OPS_kernels[1].time += __t1 - __t2;
    }
#ifndef OPS_LAZY
    ops_set_dirtybit_device(args, 8);
    ops_set_halo_dirtybit3(&args[1], range);
    ops_set_halo_dirtybit3(&args[2], range);
    ops_set_halo_dirtybit3(&args[3], range);
    ops_set_halo_dirtybit3(&args[4], range);
    ops_set_halo_dirtybit3(&args[5], range);
    ops_set_halo_dirtybit3(&args[6], range);
    ops_set_halo_dirtybit3(&args[7], range);
#endif

    if (block->instance->OPS_diags > 1) {
        //  ====================
        //  Update kernel record
        //  ====================
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[1].mpi_time += __t2 - __t1;
        block->instance->OPS_kernels[1].transfer += ops_compute_transfer(dim, start, end, &arg0);
        block->instance->OPS_kernels[1].transfer += ops_compute_transfer(dim, start, end, &arg1);
        block->instance->OPS_kernels[1].transfer += ops_compute_transfer(dim, start, end, &arg2);
        block->instance->OPS_kernels[1].transfer += ops_compute_transfer(dim, start, end, &arg3);
        block->instance->OPS_kernels[1].transfer += ops_compute_transfer(dim, start, end, &arg4);
        block->instance->OPS_kernels[1].transfer += ops_compute_transfer(dim, start, end, &arg5);
        block->instance->OPS_kernels[1].transfer += ops_compute_transfer(dim, start, end, &arg6);
        block->instance->OPS_kernels[1].transfer += ops_compute_transfer(dim, start, end, &arg7);
    }
}

#ifdef OPS_LAZY
void ops_par_loop_init_kernel(
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
    ops_arg arg7
)
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

    create_kerneldesc_and_enque("init_kernel", args, 8, 1, dim, 1, range, block, ops_par_loop_init_kernel_execute);
}
#endif
