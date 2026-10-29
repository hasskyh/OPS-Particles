// Auto-generated at 2026-10-29 00:13:46.570817 by ops-translator legacy

//header
#define OPS_2D
#define OPS_API 2
#include "ops_lib_core.h"
#ifdef OPS_MPI
#include "ops_mpi_core.h"
#include <limits>
#endif
#include "user_types.h"

#include "ops_sycl_rt_support.h"
#include "ops_sycl_reduction.h"
// global constants
cl::sycl::buffer<double,1> *dx_p=nullptr;
extern double dx;
cl::sycl::buffer<double,1> *dy_p=nullptr;
extern double dy;

void ops_init_backend() {}

void ops_decl_const_char(OPS_instance *instance, int dim, char const * type, int size, char * dat, char const * name ) {
    if (!strcmp(name,"dx")) {
        if (dx_p == nullptr) dx_p = new cl::sycl::buffer<double,1>(cl::sycl::range<1>(dim));
        auto accessor = (*dx_p).get_access<cl::sycl::access::mode::write>();
        for (int d = 0; d < dim; d++) {
            accessor[d] = ((double*)dat)[d];
        }
    }
    else
    if (!strcmp(name,"dy")) {
        if (dy_p == nullptr) dy_p = new cl::sycl::buffer<double,1>(cl::sycl::range<1>(dim));
        auto accessor = (*dy_p).get_access<cl::sycl::access::mode::write>();
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
#include "poisson_kernel_populate_kernel.cpp"
#include "poisson_kernel_update_kernel.cpp"
#include "poisson_kernel_initialguess_kernel.cpp"
#include "poisson_kernel_stencil_kernel.cpp"
#include "poisson_kernel_error_kernel.cpp"
