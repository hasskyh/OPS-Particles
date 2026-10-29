// Auto-generated at 2026-10-29 00:14:13.312536 by ops-translator legacy


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

void Ker_Diffuse_Refl_Norm_ypos(ptrm_double f, 
  const __global double * restrict u_w) {

  double u = u_w[0];
  double v = u_w[1];

  double sqrt3 = sqrt(3);

  double f2 = OPS_ACCM(f, 2, 0, 0);
  double f5 = OPS_ACCM(f, 5, 0, 0);
  double f6 = OPS_ACCM(f, 6, 0, 0);

  double rho_w = 6. * (f2 + f5 + f6) / (v * v + sqrt3 * v + 1);

  OPS_ACCM(f, 4, 0, 0) = f2 - 2. * rho_w * v / (3. * sqrt3);
  OPS_ACCM(f, 8, 0, 0) = f6 + rho_w * (u - v) / (6. * sqrt3);
  OPS_ACCM(f, 7, 0, 0) = f5 - rho_w * (u + v) / (6. * sqrt3);
  OPS_ACCM(f, 1, 0, 0) = ((2. + 2. * sqrt3 * u + 2. * u * u - v * v) * rho_w) / 18.;
  OPS_ACCM(f, 3, 0, 0) = -((-2 + 2 * sqrt3 * u - 2. * u * u + v * v) * rho_w) / 18;
  OPS_ACCM(f, 0, 0, 0) = 2. * rho_w * (2. - u * u - v * v) / 9.;
}


__kernel void ops_Ker_Diffuse_Refl_Norm_ypos(
__global double* restrict arg0,
__global const double* restrict arg1,
const int base0,
const int size0,
const int size1 ){


  int idx_y = get_global_id(1);
  int idx_x = get_global_id(0);

  if (idx_x < size0 && idx_y < size1) {
      #ifdef OPS_SOA
      ptrm_double ptr0 = { &arg0[base0 + idx_x * 1*9 + idx_y * 1*9 * xdim0_Ker_Diffuse_Refl_Norm_ypos], xdim0_Ker_Diffuse_Refl_Norm_ypos, ydim0_Ker_Diffuse_Refl_Norm_ypos};
      #else
      ptrm_double ptr0 = { &arg0[base0 + idx_x * 1*9 + idx_y * 1*9 * xdim0_Ker_Diffuse_Refl_Norm_ypos], xdim0_Ker_Diffuse_Refl_Norm_ypos, 9};
      #endif
      Ker_Diffuse_Refl_Norm_ypos(ptr0,
                                    arg1);
  }

}
