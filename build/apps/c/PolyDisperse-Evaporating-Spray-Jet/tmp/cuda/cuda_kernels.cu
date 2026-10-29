// Auto-generated at 2026-10-29 00:14:13.722737 by ops-translator legacy

//header
#define OPS_API 2
#define OPS_3D
#include "ops_lib_core.h"

#include "ops_cuda_rt_support.h"
#include "ops_cuda_reduction.h"

#include <cuComplex.h>

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
      cutilSafeCall(instance->ostream(),cudaMemcpyToSymbol(xmax, dat, dim*size));
  }
  else
  if (!strcmp(name,"ymax")) {
      cutilSafeCall(instance->ostream(),cudaMemcpyToSymbol(ymax, dat, dim*size));
  }
  else
  if (!strcmp(name,"zmax")) {
      cutilSafeCall(instance->ostream(),cudaMemcpyToSymbol(zmax, dat, dim*size));
  }
  else
  {
    throw OPSException(OPS_RUNTIME_ERROR, "error: unknown const name");
}
}


//user kernel files
#include "set_zero_kernel.cu"
