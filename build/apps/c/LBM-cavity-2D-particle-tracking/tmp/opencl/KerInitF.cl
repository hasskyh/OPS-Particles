// Auto-generated at 2026-10-29 00:14:13.370711 by ops-translator legacy


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

void KerInitF(ptrm_double  f,   
  const ptr_double  rho,   
  const ptr_double  u,   
  const ptr_double  v,   
  const __global int * restrict cx,   
  const __global int * restrict cy,   
  const __global double * restrict wi,   
  const double *cs) {
  double rhoF = OPS_ACCS(rho, 0, 0);
  double uF = OPS_ACCS(u, 0, 0);
  double vF = OPS_ACCS(v, 0, 0);
  for (int i = 0; i < 9; i++)
    OPS_ACCM(f, i, 0, 0) = calculate_feq2D(rhoF, uF, vF, cx[i], cy[i], wi[i], *cs);
}


__kernel void ops_KerInitF(
__global double* restrict arg0,
__global const double* restrict arg1,
__global const double* restrict arg2,
__global const double* restrict arg3,
__global const int* restrict arg4,
__global const int* restrict arg5,
__global const double* restrict arg6,
const double arg7,
const int base0,
const int base1,
const int base2,
const int base3,
const int size0,
const int size1 ){


  int idx_y = get_global_id(1);
  int idx_x = get_global_id(0);

  if (idx_x < size0 && idx_y < size1) {
      #ifdef OPS_SOA
      ptrm_double ptr0 = { &arg0[base0 + idx_x * 1*9 + idx_y * 1*9 * xdim0_KerInitF], xdim0_KerInitF, ydim0_KerInitF};
      #else
      ptrm_double ptr0 = { &arg0[base0 + idx_x * 1*9 + idx_y * 1*9 * xdim0_KerInitF], xdim0_KerInitF, 9};
      #endif
      const ptr_double ptr1 = { &arg1[base1 + idx_x * 1*1 + idx_y * 1*1 * xdim1_KerInitF], xdim1_KerInitF};
      const ptr_double ptr2 = { &arg2[base2 + idx_x * 1*1 + idx_y * 1*1 * xdim2_KerInitF], xdim2_KerInitF};
      const ptr_double ptr3 = { &arg3[base3 + idx_x * 1*1 + idx_y * 1*1 * xdim3_KerInitF], xdim3_KerInitF};
      KerInitF(ptr0,
                  ptr1,
                  ptr2,
                  ptr3,
                  arg4,
                  arg5,
                  arg6,
                  &arg7);
  }

}
