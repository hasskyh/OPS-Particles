// Auto-generated at 2026-10-29 00:14:01.701410 by ops-translator legacy

//header
#define OPS_API 2
#define OPS_3D
#include "ops_lib_core.h"

#include "ops_cuda_rt_support.h"
#include "ops_cuda_reduction.h"

#include <cuComplex.h>

#define OPS_FUN_PREFIX __device__ __host__
#include "user_types.h"
#ifdef OPS_MPI
#include "ops_mpi_core.h"
#endif
// global constants
__constant__ double g_small;
__constant__ double g_big;
__constant__ double dtc_safe;
__constant__ double dtu_safe;
__constant__ double dtv_safe;
__constant__ double dtw_safe;
__constant__ double dtdiv_safe;
__constant__ field_type field;
__constant__ grid_type grid;
__constant__ state_type *states;
__constant__ int number_of_states;
__constant__ int g_sphe;
__constant__ int g_point;
__constant__ int g_cube;
__constant__ double dt;

void ops_init_backend() {}

void ops_decl_const_char(OPS_instance *instance, int dim, char const *type,
int size, char *dat, char const *name){
  ops_execute(instance);
  if (!strcmp(name,"g_small")) {
      cutilSafeCall(instance->ostream(),cudaMemcpyToSymbol(g_small, dat, dim*size));
  }
  else
  if (!strcmp(name,"g_big")) {
      cutilSafeCall(instance->ostream(),cudaMemcpyToSymbol(g_big, dat, dim*size));
  }
  else
  if (!strcmp(name,"dtc_safe")) {
      cutilSafeCall(instance->ostream(),cudaMemcpyToSymbol(dtc_safe, dat, dim*size));
  }
  else
  if (!strcmp(name,"dtu_safe")) {
      cutilSafeCall(instance->ostream(),cudaMemcpyToSymbol(dtu_safe, dat, dim*size));
  }
  else
  if (!strcmp(name,"dtv_safe")) {
      cutilSafeCall(instance->ostream(),cudaMemcpyToSymbol(dtv_safe, dat, dim*size));
  }
  else
  if (!strcmp(name,"dtw_safe")) {
      cutilSafeCall(instance->ostream(),cudaMemcpyToSymbol(dtw_safe, dat, dim*size));
  }
  else
  if (!strcmp(name,"dtdiv_safe")) {
      cutilSafeCall(instance->ostream(),cudaMemcpyToSymbol(dtdiv_safe, dat, dim*size));
  }
  else
  if (!strcmp(name,"field")) {
      cutilSafeCall(instance->ostream(),cudaMemcpyToSymbol(field, dat, dim*size));
  }
  else
  if (!strcmp(name,"grid")) {
      cutilSafeCall(instance->ostream(),cudaMemcpyToSymbol(grid, dat, dim*size));
  }
  else
  if (!strcmp(name,"states")) {
      char *temp; cutilSafeCall(instance->ostream(),cudaMalloc((void**)&temp,dim*size));
      cutilSafeCall(instance->ostream(),cudaMemcpy(temp,dat,dim*size,cudaMemcpyHostToDevice));
      cutilSafeCall(instance->ostream(),cudaMemcpyToSymbol(states, &temp, sizeof(char *)));
  }
  else
  if (!strcmp(name,"number_of_states")) {
      cutilSafeCall(instance->ostream(),cudaMemcpyToSymbol(number_of_states, dat, dim*size));
  }
  else
  if (!strcmp(name,"g_sphe")) {
      cutilSafeCall(instance->ostream(),cudaMemcpyToSymbol(g_sphe, dat, dim*size));
  }
  else
  if (!strcmp(name,"g_point")) {
      cutilSafeCall(instance->ostream(),cudaMemcpyToSymbol(g_point, dat, dim*size));
  }
  else
  if (!strcmp(name,"g_cube")) {
      cutilSafeCall(instance->ostream(),cudaMemcpyToSymbol(g_cube, dat, dim*size));
  }
  else
  if (!strcmp(name,"dt")) {
      cutilSafeCall(instance->ostream(),cudaMemcpyToSymbol(dt, dat, dim*size));
  }
  else
  {
    throw OPSException(OPS_RUNTIME_ERROR, "error: unknown const name");
}
}


//user kernel files
#include "PdV_kernel_predict_kernel.cu"
#include "PdV_kernel_nopredict_kernel.cu"
#include "accelerate_kernel_kernel.cu"
#include "advec_cell_kernel1_xdir_kernel.cu"
#include "advec_cell_kernel2_xdir_kernel.cu"
#include "advec_cell_kernel3_xdir_kernel.cu"
#include "advec_cell_kernel4_xdir_kernel.cu"
#include "advec_cell_kernel1_ydir_kernel.cu"
#include "advec_cell_kernel2_ydir_kernel.cu"
#include "advec_cell_kernel3_ydir_kernel.cu"
#include "advec_cell_kernel4_ydir_kernel.cu"
#include "advec_cell_kernel1_zdir_kernel.cu"
#include "advec_cell_kernel2_zdir_kernel.cu"
#include "advec_cell_kernel3_zdir_kernel.cu"
#include "advec_cell_kernel4_zdir_kernel.cu"
#include "advec_mom_kernel_x1_kernel.cu"
#include "advec_mom_kernel_z1_kernel.cu"
#include "advec_mom_kernel_x2_kernel.cu"
#include "advec_mom_kernel_y2_kernel.cu"
#include "advec_mom_kernel_x3_kernel.cu"
#include "advec_mom_kernel_z3_kernel.cu"
#include "advec_mom_kernel_mass_flux_x_kernel.cu"
#include "advec_mom_kernel_post_pre_advec_x_kernel.cu"
#include "advec_mom_kernel1_x_nonvector_kernel.cu"
#include "advec_mom_kernel2_x_kernel.cu"
#include "advec_mom_kernel_mass_flux_y_kernel.cu"
#include "advec_mom_kernel_post_pre_advec_y_kernel.cu"
#include "advec_mom_kernel1_y_nonvector_kernel.cu"
#include "advec_mom_kernel2_y_kernel.cu"
#include "advec_mom_kernel_mass_flux_z_kernel.cu"
#include "advec_mom_kernel_post_pre_advec_z_kernel.cu"
#include "advec_mom_kernel1_z_nonvector_kernel.cu"
#include "advec_mom_kernel2_z_kernel.cu"
#include "calc_dt_kernel_kernel.cu"
#include "calc_dt_kernel_min_kernel.cu"
#include "calc_dt_kernel_get_kernel.cu"
#include "calc_dt_kernel_print_kernel.cu"
#include "field_summary_kernel_kernel.cu"
#include "flux_calc_kernelx_kernel.cu"
#include "flux_calc_kernely_kernel.cu"
#include "flux_calc_kernelz_kernel.cu"
#include "generate_chunk_kernel_kernel.cu"
#include "ideal_gas_kernel_kernel.cu"
#include "initialise_chunk_kernel_xx_kernel.cu"
#include "initialise_chunk_kernel_yy_kernel.cu"
#include "initialise_chunk_kernel_zz_kernel.cu"
#include "initialise_chunk_kernel_x_kernel.cu"
#include "initialise_chunk_kernel_y_kernel.cu"
#include "initialise_chunk_kernel_z_kernel.cu"
#include "initialise_chunk_kernel_cellx_kernel.cu"
#include "initialise_chunk_kernel_celly_kernel.cu"
#include "initialise_chunk_kernel_cellz_kernel.cu"
#include "initialise_chunk_kernel_volume_kernel.cu"
#include "reset_field_kernel1_kernel.cu"
#include "reset_field_kernel2_kernel.cu"
#include "revert_kernel_kernel.cu"
#include "update_halo_kernel1_b2_kernel.cu"
#include "update_halo_kernel1_b1_kernel.cu"
#include "update_halo_kernel1_t2_kernel.cu"
#include "update_halo_kernel1_t1_kernel.cu"
#include "update_halo_kernel1_l2_kernel.cu"
#include "update_halo_kernel1_l1_kernel.cu"
#include "update_halo_kernel1_r2_kernel.cu"
#include "update_halo_kernel1_r1_kernel.cu"
#include "update_halo_kernel1_ba2_kernel.cu"
#include "update_halo_kernel1_ba1_kernel.cu"
#include "update_halo_kernel1_fr2_kernel.cu"
#include "update_halo_kernel1_fr1_kernel.cu"
#include "update_halo_kernel2_xvel_plus_4_bot_kernel.cu"
#include "update_halo_kernel2_xvel_plus_2_bot_kernel.cu"
#include "update_halo_kernel2_xvel_plus_4_top_kernel.cu"
#include "update_halo_kernel2_xvel_plus_2_top_kernel.cu"
#include "update_halo_kernel2_xvel_minus_4_left_kernel.cu"
#include "update_halo_kernel2_xvel_minus_2_left_kernel.cu"
#include "update_halo_kernel2_xvel_minus_4_right_kernel.cu"
#include "update_halo_kernel2_xvel_minus_2_right_kernel.cu"
#include "update_halo_kernel2_xvel_plus_4_back_kernel.cu"
#include "update_halo_kernel2_xvel_plus_2_back_kernel.cu"
#include "update_halo_kernel2_xvel_plus_4_front_kernel.cu"
#include "update_halo_kernel2_xvel_plus_2_front_kernel.cu"
#include "update_halo_kernel2_yvel_minus_4_bot_kernel.cu"
#include "update_halo_kernel2_yvel_minus_2_bot_kernel.cu"
#include "update_halo_kernel2_yvel_minus_4_top_kernel.cu"
#include "update_halo_kernel2_yvel_minus_2_top_kernel.cu"
#include "update_halo_kernel2_yvel_plus_4_left_kernel.cu"
#include "update_halo_kernel2_yvel_plus_2_left_kernel.cu"
#include "update_halo_kernel2_yvel_plus_4_right_kernel.cu"
#include "update_halo_kernel2_yvel_plus_2_right_kernel.cu"
#include "update_halo_kernel2_yvel_plus_4_back_kernel.cu"
#include "update_halo_kernel2_yvel_plus_2_back_kernel.cu"
#include "update_halo_kernel2_yvel_plus_4_front_kernel.cu"
#include "update_halo_kernel2_yvel_plus_2_front_kernel.cu"
#include "update_halo_kernel2_zvel_plus_4_bot_kernel.cu"
#include "update_halo_kernel2_zvel_plus_2_bot_kernel.cu"
#include "update_halo_kernel2_zvel_plus_4_top_kernel.cu"
#include "update_halo_kernel2_zvel_plus_2_top_kernel.cu"
#include "update_halo_kernel2_zvel_plus_4_left_kernel.cu"
#include "update_halo_kernel2_zvel_plus_2_left_kernel.cu"
#include "update_halo_kernel2_zvel_plus_4_right_kernel.cu"
#include "update_halo_kernel2_zvel_plus_2_right_kernel.cu"
#include "update_halo_kernel2_zvel_minus_4_back_kernel.cu"
#include "update_halo_kernel2_zvel_minus_2_back_kernel.cu"
#include "update_halo_kernel2_zvel_minus_4_front_kernel.cu"
#include "update_halo_kernel2_zvel_minus_2_front_kernel.cu"
#include "update_halo_kernel3_plus_4_a_kernel.cu"
#include "update_halo_kernel3_plus_2_a_kernel.cu"
#include "update_halo_kernel3_plus_4_b_kernel.cu"
#include "update_halo_kernel3_plus_2_b_kernel.cu"
#include "update_halo_kernel3_minus_4_a_kernel.cu"
#include "update_halo_kernel3_minus_2_a_kernel.cu"
#include "update_halo_kernel3_minus_4_b_kernel.cu"
#include "update_halo_kernel3_minus_2_b_kernel.cu"
#include "update_halo_kernel3_plus_4_back_kernel.cu"
#include "update_halo_kernel3_plus_2_back_kernel.cu"
#include "update_halo_kernel3_plus_4_front_kernel.cu"
#include "update_halo_kernel3_plus_2_front_kernel.cu"
#include "update_halo_kernel4_minus_4_a_kernel.cu"
#include "update_halo_kernel4_minus_2_a_kernel.cu"
#include "update_halo_kernel4_minus_4_b_kernel.cu"
#include "update_halo_kernel4_minus_2_b_kernel.cu"
#include "update_halo_kernel4_plus_4_a_kernel.cu"
#include "update_halo_kernel4_plus_2_a_kernel.cu"
#include "update_halo_kernel4_plus_4_b_kernel.cu"
#include "update_halo_kernel4_plus_2_b_kernel.cu"
#include "update_halo_kernel4_plus_4_back_kernel.cu"
#include "update_halo_kernel4_plus_2_back_kernel.cu"
#include "update_halo_kernel4_plus_4_front_kernel.cu"
#include "update_halo_kernel4_plus_2_front_kernel.cu"
#include "update_halo_kernel5_plus_4_a_kernel.cu"
#include "update_halo_kernel5_plus_2_a_kernel.cu"
#include "update_halo_kernel5_plus_4_b_kernel.cu"
#include "update_halo_kernel5_plus_2_b_kernel.cu"
#include "update_halo_kernel5_plus_4_left_kernel.cu"
#include "update_halo_kernel5_plus_2_left_kernel.cu"
#include "update_halo_kernel5_plus_4_right_kernel.cu"
#include "update_halo_kernel5_plus_2_right_kernel.cu"
#include "update_halo_kernel5_minus_4_back_kernel.cu"
#include "update_halo_kernel5_minus_2_back_kernel.cu"
#include "update_halo_kernel5_minus_4_front_kernel.cu"
#include "update_halo_kernel5_minus_2_front_kernel.cu"
#include "viscosity_kernel_kernel.cu"
