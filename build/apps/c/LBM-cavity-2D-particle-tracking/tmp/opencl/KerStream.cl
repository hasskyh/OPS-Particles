// Auto-generated at 2026-10-29 00:14:13.388822 by ops-translator legacy


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

void KerStream(ptrm_double  f,  
  const ptrm_double  f_work,  
  const __global int * restrict cx,  
  const __global int * restrict cy) {
  for (int i = 0; i < 9; i++)
    OPS_ACCM(f, i, 0, 0) = OPS_ACCM(f_work, i, -cx[i], -cy[i]);
}


__kernel void ops_KerStream(
__global double* restrict arg0,
__global const double* restrict arg1,
__global const int* restrict arg2,
__global const int* restrict arg3,
const int base0,
const int base1,
const int size0,
const int size1 ){


  int idx_y = get_global_id(1);
  int idx_x = get_global_id(0);

  if (idx_x < size0 && idx_y < size1) {
      #ifdef OPS_SOA
      ptrm_double ptr0 = { &arg0[base0 + idx_x * 1*9 + idx_y * 1*9 * xdim0_KerStream], xdim0_KerStream, ydim0_KerStream};
      #else
      ptrm_double ptr0 = { &arg0[base0 + idx_x * 1*9 + idx_y * 1*9 * xdim0_KerStream], xdim0_KerStream, 9};
      #endif
      #ifdef OPS_SOA
      const ptrm_double ptr1 = { &arg1[base1 + idx_x * 1*9 + idx_y * 1*9 * xdim1_KerStream], xdim1_KerStream, ydim1_KerStream};
      #else
      const ptrm_double ptr1 = { &arg1[base1 + idx_x * 1*9 + idx_y * 1*9 * xdim1_KerStream], xdim1_KerStream, 9};
      #endif
      KerStream(ptr0,
                   ptr1,
                   arg2,
                   arg3);
  }

}
