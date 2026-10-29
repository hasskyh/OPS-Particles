// Auto-generated at 2026-10-29 00:13:46.967875 by ops-translator legacy

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
__constant__ int imax;
__constant__ int jmax;
__constant__ double pi;

void ops_init_backend() {}

void ops_decl_const_char(OPS_instance *instance, int dim, char const *type,
int size, char *dat, char const *name){
  ops_execute(instance);
  if (!strcmp(name,"imax")) {
      hipSafeCall(instance->ostream(),hipMemcpyToSymbol(HIP_SYMBOL(imax), dat, dim*size));
  }
  else
  if (!strcmp(name,"jmax")) {
      hipSafeCall(instance->ostream(),hipMemcpyToSymbol(HIP_SYMBOL(jmax), dat, dim*size));
  }
  else
  if (!strcmp(name,"pi")) {
      hipSafeCall(instance->ostream(),hipMemcpyToSymbol(HIP_SYMBOL(pi), dat, dim*size));
  }
  else
  {
    throw OPSException(OPS_RUNTIME_ERROR, "error: unknown const name");
}
}


//user kernel files
#include "set_zero_kernel.cpp"
#include "left_bndcon_kernel.cpp"
#include "right_bndcon_kernel.cpp"
#include "apply_stencil_kernel.cpp"
#include "copy_kernel.cpp"
