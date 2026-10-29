// Auto-generated at 2026-10-29 00:14:13.458373 by ops-translator legacy

//header
#define OPS_2D
#define OPS_API 2
#include "ops_lib_core.h"
#ifdef OPS_MPI
#include "ops_mpi_core.h"
#include <limits>
#endif

#include "ops_sycl_rt_support.h"
#include "ops_sycl_reduction.h"
// global constants
cl::sycl::buffer<double,1> *OMEGA_p=nullptr;
extern double OMEGA;
cl::sycl::buffer<double,1> *rho0_p=nullptr;
extern double rho0;

void ops_init_backend() {}

void ops_decl_const_char(OPS_instance *instance, int dim, char const * type, int size, char * dat, char const * name ) {
    if (!strcmp(name,"OMEGA")) {
        if (OMEGA_p == nullptr) OMEGA_p = new cl::sycl::buffer<double,1>(cl::sycl::range<1>(dim));
        auto accessor = (*OMEGA_p).get_access<cl::sycl::access::mode::write>();
        for (int d = 0; d < dim; d++) {
            accessor[d] = ((double*)dat)[d];
        }
    }
    else
    if (!strcmp(name,"rho0")) {
        if (rho0_p == nullptr) rho0_p = new cl::sycl::buffer<double,1>(cl::sycl::range<1>(dim));
        auto accessor = (*rho0_p).get_access<cl::sycl::access::mode::write>();
        for (int d = 0; d < dim; d++) {
            accessor[d] = ((double*)dat)[d];
        }
    }
    else
    {
        throw OPSException(OPS_RUNTIME_ERROR, "error: unknown const name");
    }
}

//user kernel files
#include "Ker_Diffuse_Refl_Norm_xpos_kernel.cpp"
#include "Ker_Diffuse_Refl_Norm_xneg_kernel.cpp"
#include "Ker_Diffuse_Refl_Norm_ypos_kernel.cpp"
#include "Ker_Diffuse_Refl_Norm_yneg_kernel.cpp"
#include "Ker_Diffuse_Refl_Norm_xneg_yneg_kernel.cpp"
#include "Ker_Diffuse_Refl_Norm_xpos_yneg_kernel.cpp"
#include "Ker_Diffuse_Refl_xpos_ypos_kernel.cpp"
#include "Ker_Diffuse_Refl_Norm_xneg_ypos_kernel.cpp"
#include "KerInitGrid_kernel.cpp"
#include "KerInitMacros_kernel.cpp"
#include "KerInitF_kernel.cpp"
#include "KerComputeMacros_kernel.cpp"
#include "KerCollision_kernel.cpp"
#include "KerStream_kernel.cpp"
