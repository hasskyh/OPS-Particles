// Auto-generated at 2026-10-29 00:14:12.336445 by ops-translator legacy

//header
#define OPS_2D
#define OPS_API 2
#include "ops_lib_core.h"
#ifdef OPS_MPI
#include "ops_mpi_core.h"
#endif

// global constants
extern double OMEGA;
extern double rho0;
extern double deltaUX;

void ops_init_backend() {}

void ops_decl_const_char(OPS_instance *instance, int dim, char const *type,
int size, char *dat, char const *name){
  ops_execute(instance);
  if (!strcmp(name,"OMEGA")) {
      #pragma omp target enter data map(to:OMEGA)
  }
  else
  if (!strcmp(name,"rho0")) {
      #pragma omp target enter data map(to:rho0)
  }
  else
  if (!strcmp(name,"deltaUX")) {
      #pragma omp target enter data map(to:deltaUX)
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
