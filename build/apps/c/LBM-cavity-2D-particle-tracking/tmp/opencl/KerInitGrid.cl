// Auto-generated at 2026-10-29 00:14:13.355960 by ops-translator legacy


#ifdef OCL_FMA
#pragma OPENCL FP_CONTRACT ON
#else
#pragma OPENCL FP_CONTRACT OFF
#endif
#pragma OPENCL EXTENSION cl_khr_fp64:enable

#define OPS_2D
#define OPS_API 2
#define OPS_NO_GLOBALS
#include "ops_macros.h"
#include "ops_opencl_reduction.h"

#ifndef MIN
#define MIN(a,b) ((a<b) ? (a) : (b))
#endif
#ifndef MAX
#define MAX(a,b) ((a>b) ? (a) : (b))
#endif
#ifndef SIGN
#define SIGN(a,b) ((b<0.0) ? (a*(-1)) : (a))
#endif
#define OPS_READ 0
#define OPS_WRITE 1
#define OPS_RW 2
#define OPS_INC 3
#define OPS_MIN 4
#define OPS_MAX 5

//user function

void KerInitGrid(ptrm_double  xf,
  const double *dx,
  const int *idx)  {
  OPS_ACCM(xf, 0, 0, 0) = (*dx) * static_cast<double>(idx[0]);
  OPS_ACCM(xf, 1, 0, 0) = (*dx) * static_cast<double>(idx[1]);
}


__kernel void ops_KerInitGrid(
__global double* restrict arg0,
const double arg1,
const int base0,
int arg_idx0, int arg_idx1,
const int size0,
const int size1 ){


  int idx_y = get_global_id(1);
  int idx_x = get_global_id(0);

  int arg_idx[2];
  arg_idx[0] = arg_idx0+idx_x;
  arg_idx[1] = arg_idx1+idx_y;
  if (idx_x < size0 && idx_y < size1) {
      #ifdef OPS_SOA
      ptrm_double ptr0 = { &arg0[base0 + idx_x * 1*2 + idx_y * 1*2 * xdim0_KerInitGrid], xdim0_KerInitGrid, ydim0_KerInitGrid};
      #else
      ptrm_double ptr0 = { &arg0[base0 + idx_x * 1*2 + idx_y * 1*2 * xdim0_KerInitGrid], xdim0_KerInitGrid, 2};
      #endif
      KerInitGrid(ptr0,
                     &arg1,
                     arg_idx);
  }

}
