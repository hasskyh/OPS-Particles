// Auto-generated at 2026-10-29 00:13:46.503003 by ops-translator legacy

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
__constant__ double dx;
__constant__ double dy;

void ops_init_backend() {}

void ops_decl_const_char(OPS_instance *instance, int dim, char const *type,
int size, char *dat, char const *name){
  ops_execute(instance);
  if (!strcmp(name,"dx")) {
      hipSafeCall(instance->ostream(),hipMemcpyToSymbol(HIP_SYMBOL(dx), dat, dim*size));
  }
  else
  if (!strcmp(name,"dy")) {
      hipSafeCall(instance->ostream(),hipMemcpyToSymbol(HIP_SYMBOL(dy), dat, dim*size));
  }
  else
  {
    throw OPSException(OPS_RUNTIME_ERROR, "error: unknown const name");
}
}


//user kernel files
#include "poisson_kernel_populate_kernel.cpp"
#include "poisson_kernel_update_kernel.cpp"
#include "poisson_kernel_initialguess_kernel.cpp"
#include "poisson_kernel_stencil_kernel.cpp"
#include "poisson_kernel_error_kernel.cpp"
