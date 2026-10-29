// Auto-generated at 2026-10-29 00:14:11.593990 by ops-translator legacy

//header
#define OPS_API 2
#define OPS_1D
#include "ops_lib_core.h"

#include "ops_hip_rt_support.h"
#include "ops_hip_reduction.h"


#ifdef OPS_MPI
#include "ops_mpi_core.h"
#endif
// global constants
__constant__ int nxp;
__constant__ int nyp;
__constant__ int xhalo;
__constant__ double xmin;
__constant__ double xmax;
__constant__ double dx;
__constant__ double pl;
__constant__ double pr;
__constant__ double rhol;
__constant__ double rhor;
__constant__ double ul2;
__constant__ double ur;
__constant__ double gam;
__constant__ double gam1;
__constant__ double eps;
__constant__ double lambda;
__constant__ double dt;
__constant__ double del2;
__constant__ double akap2;
__constant__ double tvdsmu;
__constant__ double con;
__constant__ double Mach;
__constant__ double xt;
__constant__ int scale;

void ops_init_backend() {}

void ops_decl_const_char(OPS_instance *instance, int dim, char const *type,
int size, char *dat, char const *name){
  ops_execute(instance);
  if (!strcmp(name,"nxp")) {
      hipSafeCall(instance->ostream(),hipMemcpyToSymbol(HIP_SYMBOL(nxp), dat, dim*size));
  }
  else
  if (!strcmp(name,"nyp")) {
      hipSafeCall(instance->ostream(),hipMemcpyToSymbol(HIP_SYMBOL(nyp), dat, dim*size));
  }
  else
  if (!strcmp(name,"xhalo")) {
      hipSafeCall(instance->ostream(),hipMemcpyToSymbol(HIP_SYMBOL(xhalo), dat, dim*size));
  }
  else
  if (!strcmp(name,"xmin")) {
      hipSafeCall(instance->ostream(),hipMemcpyToSymbol(HIP_SYMBOL(xmin), dat, dim*size));
  }
  else
  if (!strcmp(name,"xmax")) {
      hipSafeCall(instance->ostream(),hipMemcpyToSymbol(HIP_SYMBOL(xmax), dat, dim*size));
  }
  else
  if (!strcmp(name,"dx")) {
      hipSafeCall(instance->ostream(),hipMemcpyToSymbol(HIP_SYMBOL(dx), dat, dim*size));
  }
  else
  if (!strcmp(name,"pl")) {
      hipSafeCall(instance->ostream(),hipMemcpyToSymbol(HIP_SYMBOL(pl), dat, dim*size));
  }
  else
  if (!strcmp(name,"pr")) {
      hipSafeCall(instance->ostream(),hipMemcpyToSymbol(HIP_SYMBOL(pr), dat, dim*size));
  }
  else
  if (!strcmp(name,"rhol")) {
      hipSafeCall(instance->ostream(),hipMemcpyToSymbol(HIP_SYMBOL(rhol), dat, dim*size));
  }
  else
  if (!strcmp(name,"rhor")) {
      hipSafeCall(instance->ostream(),hipMemcpyToSymbol(HIP_SYMBOL(rhor), dat, dim*size));
  }
  else
  if (!strcmp(name,"ul2")) {
      hipSafeCall(instance->ostream(),hipMemcpyToSymbol(HIP_SYMBOL(ul2), dat, dim*size));
  }
  else
  if (!strcmp(name,"ur")) {
      hipSafeCall(instance->ostream(),hipMemcpyToSymbol(HIP_SYMBOL(ur), dat, dim*size));
  }
  else
  if (!strcmp(name,"gam")) {
      hipSafeCall(instance->ostream(),hipMemcpyToSymbol(HIP_SYMBOL(gam), dat, dim*size));
  }
  else
  if (!strcmp(name,"gam1")) {
      hipSafeCall(instance->ostream(),hipMemcpyToSymbol(HIP_SYMBOL(gam1), dat, dim*size));
  }
  else
  if (!strcmp(name,"eps")) {
      hipSafeCall(instance->ostream(),hipMemcpyToSymbol(HIP_SYMBOL(eps), dat, dim*size));
  }
  else
  if (!strcmp(name,"lambda")) {
      hipSafeCall(instance->ostream(),hipMemcpyToSymbol(HIP_SYMBOL(lambda), dat, dim*size));
  }
  else
  if (!strcmp(name,"dt")) {
      hipSafeCall(instance->ostream(),hipMemcpyToSymbol(HIP_SYMBOL(dt), dat, dim*size));
  }
  else
  if (!strcmp(name,"del2")) {
      hipSafeCall(instance->ostream(),hipMemcpyToSymbol(HIP_SYMBOL(del2), dat, dim*size));
  }
  else
  if (!strcmp(name,"akap2")) {
      hipSafeCall(instance->ostream(),hipMemcpyToSymbol(HIP_SYMBOL(akap2), dat, dim*size));
  }
  else
  if (!strcmp(name,"tvdsmu")) {
      hipSafeCall(instance->ostream(),hipMemcpyToSymbol(HIP_SYMBOL(tvdsmu), dat, dim*size));
  }
  else
  if (!strcmp(name,"con")) {
      hipSafeCall(instance->ostream(),hipMemcpyToSymbol(HIP_SYMBOL(con), dat, dim*size));
  }
  else
  if (!strcmp(name,"Mach")) {
      hipSafeCall(instance->ostream(),hipMemcpyToSymbol(HIP_SYMBOL(Mach), dat, dim*size));
  }
  else
  if (!strcmp(name,"xt")) {
      hipSafeCall(instance->ostream(),hipMemcpyToSymbol(HIP_SYMBOL(xt), dat, dim*size));
  }
  else
  if (!strcmp(name,"scale")) {
      hipSafeCall(instance->ostream(),hipMemcpyToSymbol(HIP_SYMBOL(scale), dat, dim*size));
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
