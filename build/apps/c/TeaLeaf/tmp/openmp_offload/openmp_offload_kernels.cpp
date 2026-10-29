// Auto-generated at 2026-10-29 00:14:09.122385 by ops-translator legacy

//header
#define OPS_2D
#define OPS_API 2
#include "ops_lib_core.h"
#ifdef OPS_MPI
#include "ops_mpi_core.h"
#endif
#include "user_types.h"

// global constants
extern field_type field;
extern grid_type grid;
extern int number_of_states;
extern state_type *states;
extern int g_circ;
extern int g_point;
extern int g_rect;

void ops_init_backend() {}

void ops_decl_const_char(OPS_instance *instance, int dim, char const *type,
int size, char *dat, char const *name){
  ops_execute(instance);
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
  {
    throw OPSException(OPS_RUNTIME_ERROR, "error: unknown const name");
}
}

//user kernel files
#include "field_summary_kernel_kernel.cpp"
#include "generate_chunk_kernel_kernel.cpp"
#include "initialise_chunk_kernel_zero_kernel.cpp"
#include "initialise_chunk_kernel_zero_x_kernel.cpp"
#include "initialise_chunk_kernel_zero_y_kernel.cpp"
#include "initialise_chunk_kernel_xx_kernel.cpp"
#include "initialise_chunk_kernel_yy_kernel.cpp"
#include "initialise_chunk_kernel_x_kernel.cpp"
#include "initialise_chunk_kernel_y_kernel.cpp"
#include "initialise_chunk_kernel_cellx_kernel.cpp"
#include "initialise_chunk_kernel_celly_kernel.cpp"
#include "initialise_chunk_kernel_volume_kernel.cpp"
#include "set_field_kernel_kernel.cpp"
#include "tea_leaf_init_zero2_kernel_kernel.cpp"
#include "tea_leaf_yeqx_kernel_kernel.cpp"
#include "tea_leaf_dot_kernel_kernel.cpp"
#include "tea_leaf_cg_calc_w_reduce_kernel_kernel.cpp"
#include "tea_leaf_axpy_kernel_kernel.cpp"
#include "tea_leaf_cg_calc_ur_r_reduce_kernel_kernel.cpp"
#include "tea_leaf_axpby_kernel_kernel.cpp"
#include "tea_leaf_cheby_init_kernel_kernel.cpp"
#include "tea_leaf_recip3_kernel_kernel.cpp"
#include "tea_leaf_xpy_kernel_kernel.cpp"
#include "tea_leaf_common_init_u_u0_kernel_kernel.cpp"
#include "tea_leaf_recip_kernel_kernel.cpp"
#include "tea_leaf_common_init_Kx_Ky_kernel_kernel.cpp"
#include "tea_leaf_init_zero_kernel_kernel.cpp"
#include "tea_leaf_common_init_kernel_kernel.cpp"
#include "tea_leaf_recip2_kernel_kernel.cpp"
#include "tea_leaf_common_residual_kernel_kernel.cpp"
#include "tea_leaf_norm2_kernel_kernel.cpp"
#include "tea_leaf_common_init_diag_init_kernel_kernel.cpp"
#include "tea_leaf_zeqxty_kernel_kernel.cpp"
#include "tea_leaf_jacobi_kernel_kernel.cpp"
#include "tea_leaf_ppcg_init1_kernel_kernel.cpp"
#include "tea_leaf_ppcg_init2_kernel_kernel.cpp"
#include "tea_leaf_ppcg_inner1_kernel_kernel.cpp"
#include "tea_leaf_ppcg_inner2_kernel_kernel.cpp"
#include "tea_leaf_ppcg_reduce_kernel_kernel.cpp"
#include "update_halo_kernel1_b2_kernel.cpp"
#include "update_halo_kernel1_b1_kernel.cpp"
#include "update_halo_kernel1_t2_kernel.cpp"
#include "update_halo_kernel1_t1_kernel.cpp"
#include "update_halo_kernel1_l2_kernel.cpp"
#include "update_halo_kernel1_l1_kernel.cpp"
#include "update_halo_kernel1_r2_kernel.cpp"
#include "update_halo_kernel1_r1_kernel.cpp"
