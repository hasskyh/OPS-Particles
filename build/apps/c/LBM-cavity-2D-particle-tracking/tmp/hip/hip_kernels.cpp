// Auto-generated at 2026-10-29 00:14:13.288671 by ops-translator legacy

//header
#define OPS_API 2
#define OPS_2D
#include "ops_lib_core.h"

#include "ops_hip_rt_support.h"
#include "ops_hip_reduction.h"


#ifdef OPS_MPI
#include "ops_mpi_core.h"
#endif
// global constants
__constant__ double OMEGA;
__constant__ double rho0;

void ops_init_backend() {}

void ops_decl_const_char(OPS_instance *instance, int dim, char const *type,
int size, char *dat, char const *name){
  ops_execute(instance);
  if (!strcmp(name,"OMEGA")) {
      hipSafeCall(instance->ostream(),hipMemcpyToSymbol(HIP_SYMBOL(OMEGA), dat, dim*size));
  }
  else
  if (!strcmp(name,"rho0")) {
      hipSafeCall(instance->ostream(),hipMemcpyToSymbol(HIP_SYMBOL(rho0), dat, dim*size));
  }
  else
  {
    throw OPSException(OPS_RUNTIME_ERROR, "error: unknown const name");
}
}


//user kernel files
#include "Ker_Diffuse_Refl_Norm_xpos_kernel.cpp"
#include "Ker_Diffuse_Refl_Norm_xneg_kernel.cpp"
#include "Ker_Diffuse_Refl_Norm_ypos_kernel.cpp"
#include "Ker_Diffuse_Refl_Norm_yneg_kernel.cpp"
#include "Ker_Diffuse_Refl_Norm_xneg_yneg_kernel.cpp"
#include "Ker_Diffuse_Refl_Norm_xpos_yneg_kernel.cpp"
#include "Ker_Diffuse_Refl_xpos_ypos_kernel.cpp"
#include "Ker_Diffuse_Refl_Norm_xneg_ypos_kernel.cpp"
#include "KerInitGrid_kernel.cpp"
#include "KerInitMacros_kernel.cpp"
#include "KerInitF_kernel.cpp"
#include "KerComputeMacros_kernel.cpp"
#include "KerCollision_kernel.cpp"
#include "KerStream_kernel.cpp"
