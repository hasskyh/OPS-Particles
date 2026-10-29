// Auto-generated at 2026-10-29 00:14:13.760376 by ops-translator legacy

//header
#define OPS_API 2
#define OPS_3D
#include "ops_lib_core.h"

#include "ops_hip_rt_support.h"
#include "ops_hip_reduction.h"


#ifdef OPS_MPI
#include "ops_mpi_core.h"
#endif
// global constants
__constant__ int xmax;
__constant__ int ymax;
__constant__ int zmax;

void ops_init_backend() {}

void ops_decl_const_char(OPS_instance *instance, int dim, char const *type,
int size, char *dat, char const *name){
  ops_execute(instance);
  if (!strcmp(name,"xmax")) {
      hipSafeCall(instance->ostream(),hipMemcpyToSymbol(HIP_SYMBOL(xmax), dat, dim*size));
  }
  else
  if (!strcmp(name,"ymax")) {
      hipSafeCall(instance->ostream(),hipMemcpyToSymbol(HIP_SYMBOL(ymax), dat, dim*size));
  }
  else
  if (!strcmp(name,"zmax")) {
      hipSafeCall(instance->ostream(),hipMemcpyToSymbol(HIP_SYMBOL(zmax), dat, dim*size));
  }
  else
  {
    throw OPSException(OPS_RUNTIME_ERROR, "error: unknown const name");
}
}


//user kernel files
#include "set_zero_kernel.cpp"
