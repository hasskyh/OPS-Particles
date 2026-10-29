// Auto-generated at 2026-10-29 00:14:10.112314 by ops-translator legacy

//header
#define OPS_API 2
#define OPS_2D
#include "ops_lib_core.h"

#include "ops_hip_rt_support.h"
#include "ops_hip_reduction.h"


#define OPS_FUN_PREFIX __device__ __host__
#include "user_types.h"
#ifdef OPS_MPI
#include "ops_mpi_core.h"
#endif
// global constants
__constant__ field_type field;
__constant__ grid_type grid;
__constant__ int number_of_states;
__constant__ state_type *states;
__constant__ int g_circ;
__constant__ int g_point;
__constant__ int g_rect;

void ops_init_backend() {}

void ops_decl_const_char(OPS_instance *instance, int dim, char const *type,
int size, char *dat, char const *name){
  ops_execute(instance);
  if (!strcmp(name,"field")) {
      hipSafeCall(instance->ostream(),hipMemcpyToSymbol(HIP_SYMBOL(field), dat, dim*size));
  }
  else
  if (!strcmp(name,"grid")) {
      hipSafeCall(instance->ostream(),hipMemcpyToSymbol(HIP_SYMBOL(grid), dat, dim*size));
  }
  else
  if (!strcmp(name,"number_of_states")) {
      hipSafeCall(instance->ostream(),hipMemcpyToSymbol(HIP_SYMBOL(number_of_states), dat, dim*size));
  }
  else
  if (!strcmp(name,"states")) {
      char *temp; hipSafeCall(instance->ostream(),hipMalloc((void**)&temp,dim*size));
      hipSafeCall(instance->ostream(),hipMemcpy(temp,dat,dim*size,hipMemcpyHostToDevice));
      hipSafeCall(instance->ostream(),hipMemcpyToSymbol(HIP_SYMBOL(states), &temp, sizeof(char *)));
  }
  else
  if (!strcmp(name,"g_circ")) {
      hipSafeCall(instance->ostream(),hipMemcpyToSymbol(HIP_SYMBOL(g_circ), dat, dim*size));
  }
  else
  if (!strcmp(name,"g_point")) {
      hipSafeCall(instance->ostream(),hipMemcpyToSymbol(HIP_SYMBOL(g_point), dat, dim*size));
  }
  else
  if (!strcmp(name,"g_rect")) {
      hipSafeCall(instance->ostream(),hipMemcpyToSymbol(HIP_SYMBOL(g_rect), dat, dim*size));
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
