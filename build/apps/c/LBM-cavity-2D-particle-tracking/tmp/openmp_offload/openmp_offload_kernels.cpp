// Auto-generated at 2026-10-29 00:14:13.156822 by ops-translator legacy

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
  {
    throw OPSException(OPS_RUNTIME_ERROR, "error: unknown const name");
}
}

//user kernel files
#include "Ker_Diffuse_Refl_Norm_xpos_kernel.cpp"
#include "Ker_Diffuse_Refl_Norm_xneg_kernel.cpp"
#include "Ker_Diffuse_Refl_Norm_ypos_kernel.cpp"
#include "Ker_Diffuse_Refl_Norm_yneg_kernel.cpp"
#include "Ker_Diffuse_Refl_Norm_xneg_yneg_kernel.cpp"
#include "Ker_Diffuse_Refl_Norm_xpos_yneg_kernel.cpp"
#include "Ker_Diffuse_Refl_xpos_ypos_kernel.cpp"
#include "Ker_Diffuse_Refl_Norm_xneg_ypos_kernel.cpp"
#include "KerInitGrid_kernel.cpp"
#include "KerInitMacros_kernel.cpp"
#include "KerInitF_kernel.cpp"
#include "KerComputeMacros_kernel.cpp"
#include "KerCollision_kernel.cpp"
#include "KerStream_kernel.cpp"
