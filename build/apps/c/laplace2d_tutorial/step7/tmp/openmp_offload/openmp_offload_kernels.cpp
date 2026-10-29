// Auto-generated at 2026-10-29 00:13:46.928969 by ops-translator legacy

//header
#define OPS_2D
#define OPS_API 2
#include "ops_lib_core.h"
#ifdef OPS_MPI
#include "ops_mpi_core.h"
#endif

// global constants
extern int imax;
extern int jmax;
extern double pi;

void ops_init_backend() {}

void ops_decl_const_char(OPS_instance *instance, int dim, char const *type,
int size, char *dat, char const *name){
  ops_execute(instance);
  if (!strcmp(name,"imax")) {
      #pragma omp target enter data map(to:imax)
  }
  else
  if (!strcmp(name,"jmax")) {
      #pragma omp target enter data map(to:jmax)
  }
  else
  if (!strcmp(name,"pi")) {
      #pragma omp target enter data map(to:pi)
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
