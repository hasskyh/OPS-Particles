// Auto-generated at 2026-10-29 00:13:46.458802 by ops-translator legacy

//header
#define OPS_2D
#define OPS_API 2
#include "ops_lib_core.h"
#ifdef OPS_MPI
#include "ops_mpi_core.h"
#endif
#include "user_types.h"

// global constants
extern double dx;
extern double dy;

void ops_init_backend() {}

void ops_decl_const_char(OPS_instance *instance, int dim, char const *type,
int size, char *dat, char const *name){
  ops_execute(instance);
  if (!strcmp(name,"dx")) {
      #pragma omp target enter data map(to:dx)
  }
  else
  if (!strcmp(name,"dy")) {
      #pragma omp target enter data map(to:dy)
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
