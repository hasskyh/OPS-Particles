// Auto-generated at 2026-10-29 00:14:13.713572 by ops-translator legacy

//header
#define OPS_3D
#define OPS_API 2
#include "ops_lib_core.h"
#ifdef OPS_MPI
#include "ops_mpi_core.h"
#endif

// global constants
extern int xmax;
extern int ymax;
extern int zmax;

void ops_init_backend() {}

void ops_decl_const_char(OPS_instance *instance, int dim, char const *type,
int size, char *dat, char const *name){
  ops_execute(instance);
  if (!strcmp(name,"xmax")) {
      #pragma omp target enter data map(to:xmax)
  }
  else
  if (!strcmp(name,"ymax")) {
      #pragma omp target enter data map(to:ymax)
  }
  else
  if (!strcmp(name,"zmax")) {
      #pragma omp target enter data map(to:zmax)
  }
  else
  {
    throw OPSException(OPS_RUNTIME_ERROR, "error: unknown const name");
}
}

//user kernel files
#include "set_zero_kernel.cpp"
