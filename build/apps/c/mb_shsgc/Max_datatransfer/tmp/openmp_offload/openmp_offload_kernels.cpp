// Auto-generated at 2026-10-29 00:14:11.396500 by ops-translator legacy

//header
#define OPS_1D
#define OPS_API 2
#include "ops_lib_core.h"
#ifdef OPS_MPI
#include "ops_mpi_core.h"
#endif

// global constants
extern int nxp;
extern int nyp;
extern int xhalo;
extern double xmin;
extern double xmax;
extern double dx;
extern double pl;
extern double pr;
extern double rhol;
extern double rhor;
extern double ul2;
extern double ur;
extern double gam;
extern double gam1;
extern double eps;
extern double lambda;
extern double dt;
extern double del2;
extern double akap2;
extern double tvdsmu;
extern double con;
extern double Mach;
extern double xt;
extern int scale;

void ops_init_backend() {}

void ops_decl_const_char(OPS_instance *instance, int dim, char const *type,
int size, char *dat, char const *name){
  ops_execute(instance);
  if (!strcmp(name,"nxp")) {
      #pragma omp target enter data map(to:nxp)
  }
  else
  if (!strcmp(name,"nyp")) {
      #pragma omp target enter data map(to:nyp)
  }
  else
  if (!strcmp(name,"xhalo")) {
      #pragma omp target enter data map(to:xhalo)
  }
  else
  if (!strcmp(name,"xmin")) {
      #pragma omp target enter data map(to:xmin)
  }
  else
  if (!strcmp(name,"xmax")) {
      #pragma omp target enter data map(to:xmax)
  }
  else
  if (!strcmp(name,"dx")) {
      #pragma omp target enter data map(to:dx)
  }
  else
  if (!strcmp(name,"pl")) {
      #pragma omp target enter data map(to:pl)
  }
  else
  if (!strcmp(name,"pr")) {
      #pragma omp target enter data map(to:pr)
  }
  else
  if (!strcmp(name,"rhol")) {
      #pragma omp target enter data map(to:rhol)
  }
  else
  if (!strcmp(name,"rhor")) {
      #pragma omp target enter data map(to:rhor)
  }
  else
  if (!strcmp(name,"ul2")) {
      #pragma omp target enter data map(to:ul2)
  }
  else
  if (!strcmp(name,"ur")) {
      #pragma omp target enter data map(to:ur)
  }
  else
  if (!strcmp(name,"gam")) {
      #pragma omp target enter data map(to:gam)
  }
  else
  if (!strcmp(name,"gam1")) {
      #pragma omp target enter data map(to:gam1)
  }
  else
  if (!strcmp(name,"eps")) {
      #pragma omp target enter data map(to:eps)
  }
  else
  if (!strcmp(name,"lambda")) {
      #pragma omp target enter data map(to:lambda)
  }
  else
  if (!strcmp(name,"dt")) {
      #pragma omp target enter data map(to:dt)
  }
  else
  if (!strcmp(name,"del2")) {
      #pragma omp target enter data map(to:del2)
  }
  else
  if (!strcmp(name,"akap2")) {
      #pragma omp target enter data map(to:akap2)
  }
  else
  if (!strcmp(name,"tvdsmu")) {
      #pragma omp target enter data map(to:tvdsmu)
  }
  else
  if (!strcmp(name,"con")) {
      #pragma omp target enter data map(to:con)
  }
  else
  if (!strcmp(name,"Mach")) {
      #pragma omp target enter data map(to:Mach)
  }
  else
  if (!strcmp(name,"xt")) {
      #pragma omp target enter data map(to:xt)
  }
  else
  if (!strcmp(name,"scale")) {
      #pragma omp target enter data map(to:scale)
  }
  else
  {
    throw OPSException(OPS_RUNTIME_ERROR, "error: unknown const name");
}
}

//user kernel files
#include "gridgen_kernel_kernel.cpp"
#include "init_kernel_kernel.cpp"
#include "save_kernel_kernel.cpp"
#include "calvar_kernel_kernel.cpp"
#include "xder1_kernel_kernel.cpp"
#include "residue_eval_kernel.cpp"
#include "updateRK3_kernel_kernel.cpp"
#include "Riemann_kernel_kernel.cpp"
#include "limiter_kernel_kernel.cpp"
#include "tvd_kernel_kernel.cpp"
#include "vars_kernel_kernel.cpp"
#include "calupwindeff_kernel_kernel.cpp"
#include "fact_kernel_kernel.cpp"
#include "update_kernel_kernel.cpp"
#include "checkop_kernel_kernel.cpp"
