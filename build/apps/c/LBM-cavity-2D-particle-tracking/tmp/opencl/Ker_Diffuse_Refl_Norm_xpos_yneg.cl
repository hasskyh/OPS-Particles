// Auto-generated at 2026-10-29 00:14:13.332257 by ops-translator legacy


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

void Ker_Diffuse_Refl_Norm_xpos_yneg(ptrm_double  f, 
  const __global double * restrict u_w) {

  double u = u_w[0];
  double v = u_w[1];

  double sqrt3 = sqrt(3);

  double f1 = OPS_ACCM(f, 1, 0, 0);
  double f4 = OPS_ACCM(f, 4, 0, 0);
  double f8 = OPS_ACCM(f, 8, 0, 0);
  double rho_w = (36 * (f1 + f4 + f8)) /
                    (9. + 5. * sqrt3 * u + 3. * u * u -
                     5. * sqrt3 * v - 3. * u * v + 3. * v * v);

  OPS_ACCM(f, 6, 0, 0) = f8 + rho_w * (v - u) / (6 * sqrt3);
  OPS_ACCM(f, 3, 0, 0) = f1 - 2 * rho_w * u / (3 * sqrt3);
  OPS_ACCM(f, 2, 0, 0) = f4 + 2 * rho_w * v / (3 * sqrt3);
  OPS_ACCM(f, 0, 0, 0) = 2 * rho_w * (2 - u * u - v * v) / 9;
  OPS_ACCM(f, 5, 0, 0) = (1 + u * u + sqrt3 * v + v * v +
                             u * (sqrt3 + 3 * v))* rho_w /
                            36;
  OPS_ACCM(f, 7, 0, 0) =
      (1 - sqrt3 * u + u * u - sqrt3 * v + 3 * u * v + v * v) *
      rho_w / 36;
}


__kernel void ops_Ker_Diffuse_Refl_Norm_xpos_yneg(
__global double* restrict arg0,
__global const double* restrict arg1,
const int base0,
const int size0,
const int size1 ){


  int idx_y = get_global_id(1);
  int idx_x = get_global_id(0);

  if (idx_x < size0 && idx_y < size1) {
      #ifdef OPS_SOA
      ptrm_double ptr0 = { &arg0[base0 + idx_x * 1*9 + idx_y * 1*9 * xdim0_Ker_Diffuse_Refl_Norm_xpos_yneg], xdim0_Ker_Diffuse_Refl_Norm_xpos_yneg, ydim0_Ker_Diffuse_Refl_Norm_xpos_yneg};
      #else
      ptrm_double ptr0 = { &arg0[base0 + idx_x * 1*9 + idx_y * 1*9 * xdim0_Ker_Diffuse_Refl_Norm_xpos_yneg], xdim0_Ker_Diffuse_Refl_Norm_xpos_yneg, 9};
      #endif
      Ker_Diffuse_Refl_Norm_xpos_yneg(ptr0,
                                         arg1);
  }

}
