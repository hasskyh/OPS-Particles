// Auto-generated at 2026-10-29 00:14:13.384233 by ops-translator legacy


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

void KerCollision(ptrm_double  f_work,   
  const ptrm_double  f,   
  const ptr_double  rho,   
  const ptr_double  u,   
  const ptr_double  v,   
  const double *omega,   
  const __global int * restrict cx,   
  const __global int * restrict cy,   
  const __global double * restrict wi,   
  const double *cs) {


  double rhoF = OPS_ACCS(rho, 0, 0);
  double uF = OPS_ACCS(u, 0, 0);
  double vF = OPS_ACCS(v, 0, 0);
  for (int i = 0; i < 9; i++) {
    double feq = calculate_feq2D(rhoF, uF, vF, cx[i], cy[i], wi[i], *cs);
    OPS_ACCM(f_work, i, 0, 0) = (1. - (*omega)) * OPS_ACCM(f, i, 0, 0) + (*omega) * feq;
  }

}


__kernel void ops_KerCollision(
__global double* restrict arg0,
__global const double* restrict arg1,
__global const double* restrict arg2,
__global const double* restrict arg3,
__global const double* restrict arg4,
const double arg5,
__global const int* restrict arg6,
__global const int* restrict arg7,
__global const double* restrict arg8,
const double arg9,
const int base0,
const int base1,
const int base2,
const int base3,
const int base4,
const int size0,
const int size1 ){


  int idx_y = get_global_id(1);
  int idx_x = get_global_id(0);

  if (idx_x < size0 && idx_y < size1) {
      #ifdef OPS_SOA
      ptrm_double ptr0 = { &arg0[base0 + idx_x * 1*9 + idx_y * 1*9 * xdim0_KerCollision], xdim0_KerCollision, ydim0_KerCollision};
      #else
      ptrm_double ptr0 = { &arg0[base0 + idx_x * 1*9 + idx_y * 1*9 * xdim0_KerCollision], xdim0_KerCollision, 9};
      #endif
      #ifdef OPS_SOA
      const ptrm_double ptr1 = { &arg1[base1 + idx_x * 1*9 + idx_y * 1*9 * xdim1_KerCollision], xdim1_KerCollision, ydim1_KerCollision};
      #else
      const ptrm_double ptr1 = { &arg1[base1 + idx_x * 1*9 + idx_y * 1*9 * xdim1_KerCollision], xdim1_KerCollision, 9};
      #endif
      const ptr_double ptr2 = { &arg2[base2 + idx_x * 1*1 + idx_y * 1*1 * xdim2_KerCollision], xdim2_KerCollision};
      const ptr_double ptr3 = { &arg3[base3 + idx_x * 1*1 + idx_y * 1*1 * xdim3_KerCollision], xdim3_KerCollision};
      const ptr_double ptr4 = { &arg4[base4 + idx_x * 1*1 + idx_y * 1*1 * xdim4_KerCollision], xdim4_KerCollision};
      KerCollision(ptr0,
                      ptr1,
                      ptr2,
                      ptr3,
                      ptr4,
                      &arg5,
                      arg6,
                      arg7,
                      arg8,
                      &arg9);
  }

}
