// Auto-generated at 2026-10-29 00:14:03.428186 by ops-translator legacy

//header
#define OPS_API 2
#define OPS_3D
#include "ops_lib_core.h"

#include "ops_hip_rt_support.h"
#include "ops_hip_reduction.h"


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
      hipSafeCall(instance->ostream(),hipMemcpyToSymbol(HIP_SYMBOL(g_small), dat, dim*size));
  }
  else
  if (!strcmp(name,"g_big")) {
      hipSafeCall(instance->ostream(),hipMemcpyToSymbol(HIP_SYMBOL(g_big), dat, dim*size));
  }
  else
  if (!strcmp(name,"dtc_safe")) {
      hipSafeCall(instance->ostream(),hipMemcpyToSymbol(HIP_SYMBOL(dtc_safe), dat, dim*size));
  }
  else
  if (!strcmp(name,"dtu_safe")) {
      hipSafeCall(instance->ostream(),hipMemcpyToSymbol(HIP_SYMBOL(dtu_safe), dat, dim*size));
  }
  else
  if (!strcmp(name,"dtv_safe")) {
      hipSafeCall(instance->ostream(),hipMemcpyToSymbol(HIP_SYMBOL(dtv_safe), dat, dim*size));
  }
  else
  if (!strcmp(name,"dtw_safe")) {
      hipSafeCall(instance->ostream(),hipMemcpyToSymbol(HIP_SYMBOL(dtw_safe), dat, dim*size));
  }
  else
  if (!strcmp(name,"dtdiv_safe")) {
      hipSafeCall(instance->ostream(),hipMemcpyToSymbol(HIP_SYMBOL(dtdiv_safe), dat, dim*size));
  }
  else
  if (!strcmp(name,"field")) {
      hipSafeCall(instance->ostream(),hipMemcpyToSymbol(HIP_SYMBOL(field), dat, dim*size));
  }
  else
  if (!strcmp(name,"grid")) {
      hipSafeCall(instance->ostream(),hipMemcpyToSymbol(HIP_SYMBOL(grid), dat, dim*size));
  }
  else
  if (!strcmp(name,"states")) {
      char *temp; hipSafeCall(instance->ostream(),hipMalloc((void**)&temp,dim*size));
      hipSafeCall(instance->ostream(),hipMemcpy(temp,dat,dim*size,hipMemcpyHostToDevice));
      hipSafeCall(instance->ostream(),hipMemcpyToSymbol(HIP_SYMBOL(states), &temp, sizeof(char *)));
  }
  else
  if (!strcmp(name,"number_of_states")) {
      hipSafeCall(instance->ostream(),hipMemcpyToSymbol(HIP_SYMBOL(number_of_states), dat, dim*size));
  }
  else
  if (!strcmp(name,"g_sphe")) {
      hipSafeCall(instance->ostream(),hipMemcpyToSymbol(HIP_SYMBOL(g_sphe), dat, dim*size));
  }
  else
  if (!strcmp(name,"g_point")) {
      hipSafeCall(instance->ostream(),hipMemcpyToSymbol(HIP_SYMBOL(g_point), dat, dim*size));
  }
  else
  if (!strcmp(name,"g_cube")) {
      hipSafeCall(instance->ostream(),hipMemcpyToSymbol(HIP_SYMBOL(g_cube), dat, dim*size));
  }
  else
  if (!strcmp(name,"dt")) {
      hipSafeCall(instance->ostream(),hipMemcpyToSymbol(HIP_SYMBOL(dt), dat, dim*size));
  }
  else
  {
    throw OPSException(OPS_RUNTIME_ERROR, "error: unknown const name");
}
}


//user kernel files
#include "PdV_kernel_predict_kernel.cpp"
#include "PdV_kernel_nopredict_kernel.cpp"
#include "accelerate_kernel_kernel.cpp"
#include "advec_cell_kernel1_xdir_kernel.cpp"
#include "advec_cell_kernel2_xdir_kernel.cpp"
#include "advec_cell_kernel3_xdir_kernel.cpp"
#include "advec_cell_kernel4_xdir_kernel.cpp"
#include "advec_cell_kernel1_ydir_kernel.cpp"
#include "advec_cell_kernel2_ydir_kernel.cpp"
#include "advec_cell_kernel3_ydir_kernel.cpp"
#include "advec_cell_kernel4_ydir_kernel.cpp"
#include "advec_cell_kernel1_zdir_kernel.cpp"
#include "advec_cell_kernel2_zdir_kernel.cpp"
#include "advec_cell_kernel3_zdir_kernel.cpp"
#include "advec_cell_kernel4_zdir_kernel.cpp"
#include "advec_mom_kernel_x1_kernel.cpp"
#include "advec_mom_kernel_z1_kernel.cpp"
#include "advec_mom_kernel_x2_kernel.cpp"
#include "advec_mom_kernel_y2_kernel.cpp"
#include "advec_mom_kernel_x3_kernel.cpp"
#include "advec_mom_kernel_z3_kernel.cpp"
#include "advec_mom_kernel_mass_flux_x_kernel.cpp"
#include "advec_mom_kernel_post_pre_advec_x_kernel.cpp"
#include "advec_mom_kernel1_x_nonvector_kernel.cpp"
#include "advec_mom_kernel2_x_kernel.cpp"
#include "advec_mom_kernel_mass_flux_y_kernel.cpp"
#include "advec_mom_kernel_post_pre_advec_y_kernel.cpp"
#include "advec_mom_kernel1_y_nonvector_kernel.cpp"
#include "advec_mom_kernel2_y_kernel.cpp"
#include "advec_mom_kernel_mass_flux_z_kernel.cpp"
#include "advec_mom_kernel_post_pre_advec_z_kernel.cpp"
#include "advec_mom_kernel1_z_nonvector_kernel.cpp"
#include "advec_mom_kernel2_z_kernel.cpp"
#include "calc_dt_kernel_kernel.cpp"
#include "calc_dt_kernel_min_kernel.cpp"
#include "calc_dt_kernel_get_kernel.cpp"
#include "calc_dt_kernel_print_kernel.cpp"
#include "field_summary_kernel_kernel.cpp"
#include "flux_calc_kernelx_kernel.cpp"
#include "flux_calc_kernely_kernel.cpp"
#include "flux_calc_kernelz_kernel.cpp"
#include "generate_chunk_kernel_kernel.cpp"
#include "ideal_gas_kernel_kernel.cpp"
#include "initialise_chunk_kernel_xx_kernel.cpp"
#include "initialise_chunk_kernel_yy_kernel.cpp"
#include "initialise_chunk_kernel_zz_kernel.cpp"
#include "initialise_chunk_kernel_x_kernel.cpp"
#include "initialise_chunk_kernel_y_kernel.cpp"
#include "initialise_chunk_kernel_z_kernel.cpp"
#include "initialise_chunk_kernel_cellx_kernel.cpp"
#include "initialise_chunk_kernel_celly_kernel.cpp"
#include "initialise_chunk_kernel_cellz_kernel.cpp"
#include "initialise_chunk_kernel_volume_kernel.cpp"
#include "reset_field_kernel1_kernel.cpp"
#include "reset_field_kernel2_kernel.cpp"
#include "revert_kernel_kernel.cpp"
#include "update_halo_kernel1_b2_kernel.cpp"
#include "update_halo_kernel1_b1_kernel.cpp"
#include "update_halo_kernel1_t2_kernel.cpp"
#include "update_halo_kernel1_t1_kernel.cpp"
#include "update_halo_kernel1_l2_kernel.cpp"
#include "update_halo_kernel1_l1_kernel.cpp"
#include "update_halo_kernel1_r2_kernel.cpp"
#include "update_halo_kernel1_r1_kernel.cpp"
#include "update_halo_kernel1_ba2_kernel.cpp"
#include "update_halo_kernel1_ba1_kernel.cpp"
#include "update_halo_kernel1_fr2_kernel.cpp"
#include "update_halo_kernel1_fr1_kernel.cpp"
#include "update_halo_kernel2_xvel_plus_4_bot_kernel.cpp"
#include "update_halo_kernel2_xvel_plus_2_bot_kernel.cpp"
#include "update_halo_kernel2_xvel_plus_4_top_kernel.cpp"
#include "update_halo_kernel2_xvel_plus_2_top_kernel.cpp"
#include "update_halo_kernel2_xvel_minus_4_left_kernel.cpp"
#include "update_halo_kernel2_xvel_minus_2_left_kernel.cpp"
#include "update_halo_kernel2_xvel_minus_4_right_kernel.cpp"
#include "update_halo_kernel2_xvel_minus_2_right_kernel.cpp"
#include "update_halo_kernel2_xvel_plus_4_back_kernel.cpp"
#include "update_halo_kernel2_xvel_plus_2_back_kernel.cpp"
#include "update_halo_kernel2_xvel_plus_4_front_kernel.cpp"
#include "update_halo_kernel2_xvel_plus_2_front_kernel.cpp"
#include "update_halo_kernel2_yvel_minus_4_bot_kernel.cpp"
#include "update_halo_kernel2_yvel_minus_2_bot_kernel.cpp"
#include "update_halo_kernel2_yvel_minus_4_top_kernel.cpp"
#include "update_halo_kernel2_yvel_minus_2_top_kernel.cpp"
#include "update_halo_kernel2_yvel_plus_4_left_kernel.cpp"
#include "update_halo_kernel2_yvel_plus_2_left_kernel.cpp"
#include "update_halo_kernel2_yvel_plus_4_right_kernel.cpp"
#include "update_halo_kernel2_yvel_plus_2_right_kernel.cpp"
#include "update_halo_kernel2_yvel_plus_4_back_kernel.cpp"
#include "update_halo_kernel2_yvel_plus_2_back_kernel.cpp"
#include "update_halo_kernel2_yvel_plus_4_front_kernel.cpp"
#include "update_halo_kernel2_yvel_plus_2_front_kernel.cpp"
#include "update_halo_kernel2_zvel_plus_4_bot_kernel.cpp"
#include "update_halo_kernel2_zvel_plus_2_bot_kernel.cpp"
#include "update_halo_kernel2_zvel_plus_4_top_kernel.cpp"
#include "update_halo_kernel2_zvel_plus_2_top_kernel.cpp"
#include "update_halo_kernel2_zvel_plus_4_left_kernel.cpp"
#include "update_halo_kernel2_zvel_plus_2_left_kernel.cpp"
#include "update_halo_kernel2_zvel_plus_4_right_kernel.cpp"
#include "update_halo_kernel2_zvel_plus_2_right_kernel.cpp"
#include "update_halo_kernel2_zvel_minus_4_back_kernel.cpp"
#include "update_halo_kernel2_zvel_minus_2_back_kernel.cpp"
#include "update_halo_kernel2_zvel_minus_4_front_kernel.cpp"
#include "update_halo_kernel2_zvel_minus_2_front_kernel.cpp"
#include "update_halo_kernel3_plus_4_a_kernel.cpp"
#include "update_halo_kernel3_plus_2_a_kernel.cpp"
#include "update_halo_kernel3_plus_4_b_kernel.cpp"
#include "update_halo_kernel3_plus_2_b_kernel.cpp"
#include "update_halo_kernel3_minus_4_a_kernel.cpp"
#include "update_halo_kernel3_minus_2_a_kernel.cpp"
#include "update_halo_kernel3_minus_4_b_kernel.cpp"
#include "update_halo_kernel3_minus_2_b_kernel.cpp"
#include "update_halo_kernel3_plus_4_back_kernel.cpp"
#include "update_halo_kernel3_plus_2_back_kernel.cpp"
#include "update_halo_kernel3_plus_4_front_kernel.cpp"
#include "update_halo_kernel3_plus_2_front_kernel.cpp"
#include "update_halo_kernel4_minus_4_a_kernel.cpp"
#include "update_halo_kernel4_minus_2_a_kernel.cpp"
#include "update_halo_kernel4_minus_4_b_kernel.cpp"
#include "update_halo_kernel4_minus_2_b_kernel.cpp"
#include "update_halo_kernel4_plus_4_a_kernel.cpp"
#include "update_halo_kernel4_plus_2_a_kernel.cpp"
#include "update_halo_kernel4_plus_4_b_kernel.cpp"
#include "update_halo_kernel4_plus_2_b_kernel.cpp"
#include "update_halo_kernel4_plus_4_back_kernel.cpp"
#include "update_halo_kernel4_plus_2_back_kernel.cpp"
#include "update_halo_kernel4_plus_4_front_kernel.cpp"
#include "update_halo_kernel4_plus_2_front_kernel.cpp"
#include "update_halo_kernel5_plus_4_a_kernel.cpp"
#include "update_halo_kernel5_plus_2_a_kernel.cpp"
#include "update_halo_kernel5_plus_4_b_kernel.cpp"
#include "update_halo_kernel5_plus_2_b_kernel.cpp"
#include "update_halo_kernel5_plus_4_left_kernel.cpp"
#include "update_halo_kernel5_plus_2_left_kernel.cpp"
#include "update_halo_kernel5_plus_4_right_kernel.cpp"
#include "update_halo_kernel5_plus_2_right_kernel.cpp"
#include "update_halo_kernel5_minus_4_back_kernel.cpp"
#include "update_halo_kernel5_minus_2_back_kernel.cpp"
#include "update_halo_kernel5_minus_4_front_kernel.cpp"
#include "update_halo_kernel5_minus_2_front_kernel.cpp"
#include "viscosity_kernel_kernel.cpp"
