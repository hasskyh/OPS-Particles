// Auto-generated at 2026-10-29 00:14:12.394418 by ops-translator legacy

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
__constant__ double deltaUX;

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
  if (!strcmp(name,"deltaUX")) {
      cutilSafeCall(instance->ostream(),cudaMemcpyToSymbol(deltaUX, dat, dim*size));
  }
  else
  {
    throw OPSException(OPS_RUNTIME_ERROR, "error: unknown const name");
}
}


//user kernel files
#include "init_solid_kernel.cu"
#include "init_n_kernel.cu"
#include "timeloop_eqA_kernel.cu"
#include "timeloop_eqB_kernel.cu"
#include "timeloop_eqC_kernel.cu"
#include "timeloop_eqD_kernel.cu"
#include "timeloop_eqE_kernel.cu"
#include "timeloop_eqF_kernel.cu"
#include "timeloop_eqG_kernel.cu"
#include "timeloop_eqH_kernel.cu"
#include "timeloop_eqI_kernel.cu"
#include "timeloop_eqJ_kernel.cu"
#include "timeloop_eqK_kernel.cu"
#include "timeloop_eqL_kernel.cu"
#include "timeloop_eqM_kernel.cu"
