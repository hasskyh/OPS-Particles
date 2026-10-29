// Auto-generated at 2026-10-29 00:14:12.482817 by ops-translator legacy


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

void init_solid(ptr_int SOLID,
  const int *idx) {
    int i = idx[0], j = idx[1];

    if( (i>=2 && i<=65 && j>=4 && j<=29) || (i>=45 && i<=123 && j>=41 && j<=65) || (i>=30 && i<=101 && j>=91 && j<=115))
        OPS_ACCS(SOLID, 0,0) = 0;
    else
        OPS_ACCS(SOLID, 0,0) = 1;

}


__kernel void ops_init_solid(
__global int* restrict arg0,
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
      ptr_int ptr0 = { &arg0[base0 + idx_x * 1*1 + idx_y * 1*1 * xdim0_init_solid], xdim0_init_solid};
      init_solid(ptr0,
                    arg_idx);
  }

}
