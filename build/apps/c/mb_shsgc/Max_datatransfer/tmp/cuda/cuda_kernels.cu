// Auto-generated at 2026-10-29 00:14:11.516581 by ops-translator legacy

//header
#define OPS_API 2
#define OPS_1D
#include "ops_lib_core.h"

#include "ops_cuda_rt_support.h"
#include "ops_cuda_reduction.h"

#include <cuComplex.h>

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
      cutilSafeCall(instance->ostream(),cudaMemcpyToSymbol(nxp, dat, dim*size));
  }
  else
  if (!strcmp(name,"nyp")) {
      cutilSafeCall(instance->ostream(),cudaMemcpyToSymbol(nyp, dat, dim*size));
  }
  else
  if (!strcmp(name,"xhalo")) {
      cutilSafeCall(instance->ostream(),cudaMemcpyToSymbol(xhalo, dat, dim*size));
  }
  else
  if (!strcmp(name,"xmin")) {
      cutilSafeCall(instance->ostream(),cudaMemcpyToSymbol(xmin, dat, dim*size));
  }
  else
  if (!strcmp(name,"xmax")) {
      cutilSafeCall(instance->ostream(),cudaMemcpyToSymbol(xmax, dat, dim*size));
  }
  else
  if (!strcmp(name,"dx")) {
      cutilSafeCall(instance->ostream(),cudaMemcpyToSymbol(dx, dat, dim*size));
  }
  else
  if (!strcmp(name,"pl")) {
      cutilSafeCall(instance->ostream(),cudaMemcpyToSymbol(pl, dat, dim*size));
  }
  else
  if (!strcmp(name,"pr")) {
      cutilSafeCall(instance->ostream(),cudaMemcpyToSymbol(pr, dat, dim*size));
  }
  else
  if (!strcmp(name,"rhol")) {
      cutilSafeCall(instance->ostream(),cudaMemcpyToSymbol(rhol, dat, dim*size));
  }
  else
  if (!strcmp(name,"rhor")) {
      cutilSafeCall(instance->ostream(),cudaMemcpyToSymbol(rhor, dat, dim*size));
  }
  else
  if (!strcmp(name,"ul2")) {
      cutilSafeCall(instance->ostream(),cudaMemcpyToSymbol(ul2, dat, dim*size));
  }
  else
  if (!strcmp(name,"ur")) {
      cutilSafeCall(instance->ostream(),cudaMemcpyToSymbol(ur, dat, dim*size));
  }
  else
  if (!strcmp(name,"gam")) {
      cutilSafeCall(instance->ostream(),cudaMemcpyToSymbol(gam, dat, dim*size));
  }
  else
  if (!strcmp(name,"gam1")) {
      cutilSafeCall(instance->ostream(),cudaMemcpyToSymbol(gam1, dat, dim*size));
  }
  else
  if (!strcmp(name,"eps")) {
      cutilSafeCall(instance->ostream(),cudaMemcpyToSymbol(eps, dat, dim*size));
  }
  else
  if (!strcmp(name,"lambda")) {
      cutilSafeCall(instance->ostream(),cudaMemcpyToSymbol(lambda, dat, dim*size));
  }
  else
  if (!strcmp(name,"dt")) {
      cutilSafeCall(instance->ostream(),cudaMemcpyToSymbol(dt, dat, dim*size));
  }
  else
  if (!strcmp(name,"del2")) {
      cutilSafeCall(instance->ostream(),cudaMemcpyToSymbol(del2, dat, dim*size));
  }
  else
  if (!strcmp(name,"akap2")) {
      cutilSafeCall(instance->ostream(),cudaMemcpyToSymbol(akap2, dat, dim*size));
  }
  else
  if (!strcmp(name,"tvdsmu")) {
      cutilSafeCall(instance->ostream(),cudaMemcpyToSymbol(tvdsmu, dat, dim*size));
  }
  else
  if (!strcmp(name,"con")) {
      cutilSafeCall(instance->ostream(),cudaMemcpyToSymbol(con, dat, dim*size));
  }
  else
  if (!strcmp(name,"Mach")) {
      cutilSafeCall(instance->ostream(),cudaMemcpyToSymbol(Mach, dat, dim*size));
  }
  else
  if (!strcmp(name,"xt")) {
      cutilSafeCall(instance->ostream(),cudaMemcpyToSymbol(xt, dat, dim*size));
  }
  else
  if (!strcmp(name,"scale")) {
      cutilSafeCall(instance->ostream(),cudaMemcpyToSymbol(scale, dat, dim*size));
  }
  else
  {
    throw OPSException(OPS_RUNTIME_ERROR, "error: unknown const name");
}
}


//user kernel files
#include "gridgen_kernel_kernel.cu"
#include "init_kernel_kernel.cu"
#include "save_kernel_kernel.cu"
#include "calvar_kernel_kernel.cu"
#include "xder1_kernel_kernel.cu"
#include "residue_eval_kernel.cu"
#include "updateRK3_kernel_kernel.cu"
#include "Riemann_kernel_kernel.cu"
#include "limiter_kernel_kernel.cu"
#include "tvd_kernel_kernel.cu"
#include "vars_kernel_kernel.cu"
#include "calupwindeff_kernel_kernel.cu"
#include "fact_kernel_kernel.cu"
#include "update_kernel_kernel.cu"
#include "checkop_kernel_kernel.cu"
