// Auto-generated at 2026-10-29 00:14:13.830602 by ops-translator legacy

//header
#define OPS_3D
#define OPS_API 2
#include "ops_lib_core.h"
#ifdef OPS_MPI
#include "ops_mpi_core.h"
#include <limits>
#endif

#include "ops_sycl_rt_support.h"
#include "ops_sycl_reduction.h"
// global constants
cl::sycl::buffer<int,1> *xmax_p=nullptr;
extern int xmax;
cl::sycl::buffer<int,1> *ymax_p=nullptr;
extern int ymax;
cl::sycl::buffer<int,1> *zmax_p=nullptr;
extern int zmax;

void ops_init_backend() {}

void ops_decl_const_char(OPS_instance *instance, int dim, char const * type, int size, char * dat, char const * name ) {
    if (!strcmp(name,"xmax")) {
        if (xmax_p == nullptr) xmax_p = new cl::sycl::buffer<int,1>(cl::sycl::range<1>(dim));
        auto accessor = (*xmax_p).get_access<cl::sycl::access::mode::write>();
        for (int d = 0; d < dim; d++) {
            accessor[d] = ((int*)dat)[d];
        }
    }
    else
    if (!strcmp(name,"ymax")) {
        if (ymax_p == nullptr) ymax_p = new cl::sycl::buffer<int,1>(cl::sycl::range<1>(dim));
        auto accessor = (*ymax_p).get_access<cl::sycl::access::mode::write>();
        for (int d = 0; d < dim; d++) {
            accessor[d] = ((int*)dat)[d];
        }
    }
    else
    if (!strcmp(name,"zmax")) {
        if (zmax_p == nullptr) zmax_p = new cl::sycl::buffer<int,1>(cl::sycl::range<1>(dim));
        auto accessor = (*zmax_p).get_access<cl::sycl::access::mode::write>();
        for (int d = 0; d < dim; d++) {
            accessor[d] = ((int*)dat)[d];
        }
    }
    else
    {
        throw OPSException(OPS_RUNTIME_ERROR, "error: unknown const name");
    }
}

//user kernel files
#include "set_zero_kernel.cpp"
