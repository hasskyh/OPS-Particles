// Auto-generated at 2026-10-29 00:14:12.468851 by ops-translator legacy

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
__constant__ double deltaUX;

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
  if (!strcmp(name,"deltaUX")) {
      hipSafeCall(instance->ostream(),hipMemcpyToSymbol(HIP_SYMBOL(deltaUX), dat, dim*size));
  }
  else
  {
    throw OPSException(OPS_RUNTIME_ERROR, "error: unknown const name");
}
}


//user kernel files
#include "init_solid_kernel.cpp"
#include "init_n_kernel.cpp"
#include "timeloop_eqA_kernel.cpp"
#include "timeloop_eqB_kernel.cpp"
#include "timeloop_eqC_kernel.cpp"
#include "timeloop_eqD_kernel.cpp"
#include "timeloop_eqE_kernel.cpp"
#include "timeloop_eqF_kernel.cpp"
#include "timeloop_eqG_kernel.cpp"
#include "timeloop_eqH_kernel.cpp"
#include "timeloop_eqI_kernel.cpp"
#include "timeloop_eqJ_kernel.cpp"
#include "timeloop_eqK_kernel.cpp"
#include "timeloop_eqL_kernel.cpp"
#include "timeloop_eqM_kernel.cpp"
