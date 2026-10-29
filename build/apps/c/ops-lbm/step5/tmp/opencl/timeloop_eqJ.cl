// Auto-generated at 2026-10-29 00:14:12.572703 by ops-translator legacy


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

void timeloop_eqJ(ptrm_double workArray, 
  const ptr_double rho, 
  const __global double* restrict  W) {

    for (int f = 0; f < 9; f++) {
        OPS_ACCM(workArray, f,0,0) = (1.0 + OPS_ACCM(workArray, f,0,0)) * W[f] * OPS_ACCS(rho, 0,0);
    }
}


__kernel void ops_timeloop_eqJ(
__global double* restrict arg0,
__global const double* restrict arg1,
__global const double* restrict arg2,
const int base0,
const int base1,
const int size0,
const int size1 ){


  int idx_y = get_global_id(1);
  int idx_x = get_global_id(0);

  if (idx_x < size0 && idx_y < size1) {
      #ifdef OPS_SOA
      ptrm_double ptr0 = { &arg0[base0 + idx_x * 1*9 + idx_y * 1*9 * xdim0_timeloop_eqJ], xdim0_timeloop_eqJ, ydim0_timeloop_eqJ};
      #else
      ptrm_double ptr0 = { &arg0[base0 + idx_x * 1*9 + idx_y * 1*9 * xdim0_timeloop_eqJ], xdim0_timeloop_eqJ, 9};
      #endif
      const ptr_double ptr1 = { &arg1[base1 + idx_x * 1*1 + idx_y * 1*1 * xdim1_timeloop_eqJ], xdim1_timeloop_eqJ};
      timeloop_eqJ(ptr0,
                      ptr1,
                      arg2);
  }

}
