// Auto-generated at 2026-10-29 00:13:49.095990 by ops-translator legacy

//header
#define OPS_2D
#define OPS_API 2
#include "ops_lib_core.h"
#ifdef OPS_MPI
#include "ops_mpi_core.h"
#endif
#include "user_types.h"

// global constants
extern double g_small;
extern double g_big;
extern double dtc_safe;
extern double dtu_safe;
extern double dtv_safe;
extern double dtdiv_safe;
extern field_type field;
extern grid_type grid;
extern int number_of_states;
extern state_type *states;
extern int g_circ;
extern int g_point;
extern int g_rect;
extern double dt;

void ops_init_backend() {}

void ops_decl_const_char(OPS_instance *instance, int dim, char const *type,
int size, char *dat, char const *name){
  ops_execute(instance);
  if (!strcmp(name,"g_small")) {
      #pragma omp target enter data map(to:g_small)
  }
  else
  if (!strcmp(name,"g_big")) {
      #pragma omp target enter data map(to:g_big)
  }
  else
  if (!strcmp(name,"dtc_safe")) {
      #pragma omp target enter data map(to:dtc_safe)
  }
  else
  if (!strcmp(name,"dtu_safe")) {
      #pragma omp target enter data map(to:dtu_safe)
  }
  else
  if (!strcmp(name,"dtv_safe")) {
      #pragma omp target enter data map(to:dtv_safe)
  }
  else
  if (!strcmp(name,"dtdiv_safe")) {
      #pragma omp target enter data map(to:dtdiv_safe)
  }
  else
  if (!strcmp(name,"field")) {
      #pragma omp target enter data map(to:field)
  }
  else
  if (!strcmp(name,"grid")) {
      #pragma omp target enter data map(to:grid)
  }
  else
  if (!strcmp(name,"number_of_states")) {
      #pragma omp target enter data map(to:number_of_states)
  }
  else
  if (!strcmp(name,"states")) {
      #pragma omp target enter data map(to:states[0:dim])
  }
  else
  if (!strcmp(name,"g_circ")) {
      #pragma omp target enter data map(to:g_circ)
  }
  else
  if (!strcmp(name,"g_point")) {
      #pragma omp target enter data map(to:g_point)
  }
  else
  if (!strcmp(name,"g_rect")) {
      #pragma omp target enter data map(to:g_rect)
  }
  else
  if (!strcmp(name,"dt")) {
      #pragma omp target enter data map(to:dt)
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
#include "advec_mom_kernel_x1_kernel.cpp"
#include "advec_mom_kernel_y1_kernel.cpp"
#include "advec_mom_kernel_x2_kernel.cpp"
#include "advec_mom_kernel_y2_kernel.cpp"
#include "advec_mom_kernel_mass_flux_x_kernel.cpp"
#include "advec_mom_kernel_post_pre_advec_x_kernel.cpp"
#include "advec_mom_kernel1_x_nonvector_kernel.cpp"
#include "advec_mom_kernel2_x_kernel.cpp"
#include "advec_mom_kernel_mass_flux_y_kernel.cpp"
#include "advec_mom_kernel_post_pre_advec_y_kernel.cpp"
#include "advec_mom_kernel1_y_nonvector_kernel.cpp"
#include "advec_mom_kernel2_y_kernel.cpp"
#include "calc_dt_kernel_kernel.cpp"
#include "calc_dt_kernel_min_kernel.cpp"
#include "calc_dt_kernel_get_kernel.cpp"
#include "calc_dt_kernel_print_kernel.cpp"
#include "field_summary_kernel_kernel.cpp"
#include "flux_calc_kernelx_kernel.cpp"
#include "flux_calc_kernely_kernel.cpp"
#include "generate_chunk_kernel_kernel.cpp"
#include "ideal_gas_kernel_kernel.cpp"
#include "initialise_chunk_kernel_xx_kernel.cpp"
#include "initialise_chunk_kernel_yy_kernel.cpp"
#include "initialise_chunk_kernel_x_kernel.cpp"
#include "initialise_chunk_kernel_y_kernel.cpp"
#include "initialise_chunk_kernel_cellx_kernel.cpp"
#include "initialise_chunk_kernel_celly_kernel.cpp"
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
#include "update_halo_kernel2_xvel_plus_4_a_kernel.cpp"
#include "update_halo_kernel2_xvel_plus_2_a_kernel.cpp"
#include "update_halo_kernel2_xvel_plus_4_b_kernel.cpp"
#include "update_halo_kernel2_xvel_plus_2_b_kernel.cpp"
#include "update_halo_kernel2_xvel_minus_4_a_kernel.cpp"
#include "update_halo_kernel2_xvel_minus_2_a_kernel.cpp"
#include "update_halo_kernel2_xvel_minus_4_b_kernel.cpp"
#include "update_halo_kernel2_xvel_minus_2_b_kernel.cpp"
#include "update_halo_kernel2_yvel_minus_4_a_kernel.cpp"
#include "update_halo_kernel2_yvel_minus_2_a_kernel.cpp"
#include "update_halo_kernel2_yvel_minus_4_b_kernel.cpp"
#include "update_halo_kernel2_yvel_minus_2_b_kernel.cpp"
#include "update_halo_kernel2_yvel_plus_4_a_kernel.cpp"
#include "update_halo_kernel2_yvel_plus_2_a_kernel.cpp"
#include "update_halo_kernel2_yvel_plus_4_b_kernel.cpp"
#include "update_halo_kernel2_yvel_plus_2_b_kernel.cpp"
#include "update_halo_kernel3_plus_4_a_kernel.cpp"
#include "update_halo_kernel3_plus_2_a_kernel.cpp"
#include "update_halo_kernel3_plus_4_b_kernel.cpp"
#include "update_halo_kernel3_plus_2_b_kernel.cpp"
#include "update_halo_kernel3_minus_4_a_kernel.cpp"
#include "update_halo_kernel3_minus_2_a_kernel.cpp"
#include "update_halo_kernel3_minus_4_b_kernel.cpp"
#include "update_halo_kernel3_minus_2_b_kernel.cpp"
#include "update_halo_kernel4_minus_4_a_kernel.cpp"
#include "update_halo_kernel4_minus_2_a_kernel.cpp"
#include "update_halo_kernel4_minus_4_b_kernel.cpp"
#include "update_halo_kernel4_minus_2_b_kernel.cpp"
#include "update_halo_kernel4_plus_4_a_kernel.cpp"
#include "update_halo_kernel4_plus_2_a_kernel.cpp"
#include "update_halo_kernel4_plus_4_b_kernel.cpp"
#include "update_halo_kernel4_plus_2_b_kernel.cpp"
#include "viscosity_kernel_kernel.cpp"
