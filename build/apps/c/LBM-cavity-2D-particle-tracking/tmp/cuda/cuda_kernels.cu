// Auto-generated at 2026-10-29 00:14:13.217727 by ops-translator legacy

//header
#define OPS_API 2
#define OPS_2D
#include "ops_lib_core.h"

#include "ops_cuda_rt_support.h"
#include "ops_cuda_reduction.h"

#include <cuComplex.h>

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
      cutilSafeCall(instance->ostream(),cudaMemcpyToSymbol(OMEGA, dat, dim*size));
  }
  else
  if (!strcmp(name,"rho0")) {
      cutilSafeCall(instance->ostream(),cudaMemcpyToSymbol(rho0, dat, dim*size));
  }
  else
  {
    throw OPSException(OPS_RUNTIME_ERROR, "error: unknown const name");
}
}


//user kernel files
#include "Ker_Diffuse_Refl_Norm_xpos_kernel.cu"
#include "Ker_Diffuse_Refl_Norm_xneg_kernel.cu"
#include "Ker_Diffuse_Refl_Norm_ypos_kernel.cu"
#include "Ker_Diffuse_Refl_Norm_yneg_kernel.cu"
#include "Ker_Diffuse_Refl_Norm_xneg_yneg_kernel.cu"
#include "Ker_Diffuse_Refl_Norm_xpos_yneg_kernel.cu"
#include "Ker_Diffuse_Refl_xpos_ypos_kernel.cu"
#include "Ker_Diffuse_Refl_Norm_xneg_ypos_kernel.cu"
#include "KerInitGrid_kernel.cu"
#include "KerInitMacros_kernel.cu"
#include "KerInitF_kernel.cu"
#include "KerComputeMacros_kernel.cu"
#include "KerCollision_kernel.cu"
#include "KerStream_kernel.cu"
