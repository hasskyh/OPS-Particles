// Auto-generated at 2026-10-29 00:14:11.776470 by ops-translator legacy

//header
#define OPS_1D
#define OPS_API 2
#include "ops_lib_core.h"
#ifdef OPS_MPI
#include "ops_mpi_core.h"
#include <limits>
#endif

#include "ops_sycl_rt_support.h"
#include "ops_sycl_reduction.h"
// global constants
cl::sycl::buffer<int,1> *nxp_p=nullptr;
extern int nxp;
cl::sycl::buffer<int,1> *nyp_p=nullptr;
extern int nyp;
cl::sycl::buffer<int,1> *xhalo_p=nullptr;
extern int xhalo;
cl::sycl::buffer<double,1> *xmin_p=nullptr;
extern double xmin;
cl::sycl::buffer<double,1> *xmax_p=nullptr;
extern double xmax;
cl::sycl::buffer<double,1> *dx_p=nullptr;
extern double dx;
cl::sycl::buffer<double,1> *pl_p=nullptr;
extern double pl;
cl::sycl::buffer<double,1> *pr_p=nullptr;
extern double pr;
cl::sycl::buffer<double,1> *rhol_p=nullptr;
extern double rhol;
cl::sycl::buffer<double,1> *rhor_p=nullptr;
extern double rhor;
cl::sycl::buffer<double,1> *ul2_p=nullptr;
extern double ul2;
cl::sycl::buffer<double,1> *ur_p=nullptr;
extern double ur;
cl::sycl::buffer<double,1> *gam_p=nullptr;
extern double gam;
cl::sycl::buffer<double,1> *gam1_p=nullptr;
extern double gam1;
cl::sycl::buffer<double,1> *eps_p=nullptr;
extern double eps;
cl::sycl::buffer<double,1> *lambda_p=nullptr;
extern double lambda;
cl::sycl::buffer<double,1> *dt_p=nullptr;
extern double dt;
cl::sycl::buffer<double,1> *del2_p=nullptr;
extern double del2;
cl::sycl::buffer<double,1> *akap2_p=nullptr;
extern double akap2;
cl::sycl::buffer<double,1> *tvdsmu_p=nullptr;
extern double tvdsmu;
cl::sycl::buffer<double,1> *con_p=nullptr;
extern double con;
cl::sycl::buffer<double,1> *Mach_p=nullptr;
extern double Mach;
cl::sycl::buffer<double,1> *xt_p=nullptr;
extern double xt;
cl::sycl::buffer<int,1> *scale_p=nullptr;
extern int scale;

void ops_init_backend() {}

void ops_decl_const_char(OPS_instance *instance, int dim, char const * type, int size, char * dat, char const * name ) {
    if (!strcmp(name,"nxp")) {
        if (nxp_p == nullptr) nxp_p = new cl::sycl::buffer<int,1>(cl::sycl::range<1>(dim));
        auto accessor = (*nxp_p).get_access<cl::sycl::access::mode::write>();
        for (int d = 0; d < dim; d++) {
            accessor[d] = ((int*)dat)[d];
        }
    }
    else
    if (!strcmp(name,"nyp")) {
        if (nyp_p == nullptr) nyp_p = new cl::sycl::buffer<int,1>(cl::sycl::range<1>(dim));
        auto accessor = (*nyp_p).get_access<cl::sycl::access::mode::write>();
        for (int d = 0; d < dim; d++) {
            accessor[d] = ((int*)dat)[d];
        }
    }
    else
    if (!strcmp(name,"xhalo")) {
        if (xhalo_p == nullptr) xhalo_p = new cl::sycl::buffer<int,1>(cl::sycl::range<1>(dim));
        auto accessor = (*xhalo_p).get_access<cl::sycl::access::mode::write>();
        for (int d = 0; d < dim; d++) {
            accessor[d] = ((int*)dat)[d];
        }
    }
    else
    if (!strcmp(name,"xmin")) {
        if (xmin_p == nullptr) xmin_p = new cl::sycl::buffer<double,1>(cl::sycl::range<1>(dim));
        auto accessor = (*xmin_p).get_access<cl::sycl::access::mode::write>();
        for (int d = 0; d < dim; d++) {
            accessor[d] = ((double*)dat)[d];
        }
    }
    else
    if (!strcmp(name,"xmax")) {
        if (xmax_p == nullptr) xmax_p = new cl::sycl::buffer<double,1>(cl::sycl::range<1>(dim));
        auto accessor = (*xmax_p).get_access<cl::sycl::access::mode::write>();
        for (int d = 0; d < dim; d++) {
            accessor[d] = ((double*)dat)[d];
        }
    }
    else
    if (!strcmp(name,"dx")) {
        if (dx_p == nullptr) dx_p = new cl::sycl::buffer<double,1>(cl::sycl::range<1>(dim));
        auto accessor = (*dx_p).get_access<cl::sycl::access::mode::write>();
        for (int d = 0; d < dim; d++) {
            accessor[d] = ((double*)dat)[d];
        }
    }
    else
    if (!strcmp(name,"pl")) {
        if (pl_p == nullptr) pl_p = new cl::sycl::buffer<double,1>(cl::sycl::range<1>(dim));
        auto accessor = (*pl_p).get_access<cl::sycl::access::mode::write>();
        for (int d = 0; d < dim; d++) {
            accessor[d] = ((double*)dat)[d];
        }
    }
    else
    if (!strcmp(name,"pr")) {
        if (pr_p == nullptr) pr_p = new cl::sycl::buffer<double,1>(cl::sycl::range<1>(dim));
        auto accessor = (*pr_p).get_access<cl::sycl::access::mode::write>();
        for (int d = 0; d < dim; d++) {
            accessor[d] = ((double*)dat)[d];
        }
    }
    else
    if (!strcmp(name,"rhol")) {
        if (rhol_p == nullptr) rhol_p = new cl::sycl::buffer<double,1>(cl::sycl::range<1>(dim));
        auto accessor = (*rhol_p).get_access<cl::sycl::access::mode::write>();
        for (int d = 0; d < dim; d++) {
            accessor[d] = ((double*)dat)[d];
        }
    }
    else
    if (!strcmp(name,"rhor")) {
        if (rhor_p == nullptr) rhor_p = new cl::sycl::buffer<double,1>(cl::sycl::range<1>(dim));
        auto accessor = (*rhor_p).get_access<cl::sycl::access::mode::write>();
        for (int d = 0; d < dim; d++) {
            accessor[d] = ((double*)dat)[d];
        }
    }
    else
    if (!strcmp(name,"ul2")) {
        if (ul2_p == nullptr) ul2_p = new cl::sycl::buffer<double,1>(cl::sycl::range<1>(dim));
        auto accessor = (*ul2_p).get_access<cl::sycl::access::mode::write>();
        for (int d = 0; d < dim; d++) {
            accessor[d] = ((double*)dat)[d];
        }
    }
    else
    if (!strcmp(name,"ur")) {
        if (ur_p == nullptr) ur_p = new cl::sycl::buffer<double,1>(cl::sycl::range<1>(dim));
        auto accessor = (*ur_p).get_access<cl::sycl::access::mode::write>();
        for (int d = 0; d < dim; d++) {
            accessor[d] = ((double*)dat)[d];
        }
    }
    else
    if (!strcmp(name,"gam")) {
        if (gam_p == nullptr) gam_p = new cl::sycl::buffer<double,1>(cl::sycl::range<1>(dim));
        auto accessor = (*gam_p).get_access<cl::sycl::access::mode::write>();
        for (int d = 0; d < dim; d++) {
            accessor[d] = ((double*)dat)[d];
        }
    }
    else
    if (!strcmp(name,"gam1")) {
        if (gam1_p == nullptr) gam1_p = new cl::sycl::buffer<double,1>(cl::sycl::range<1>(dim));
        auto accessor = (*gam1_p).get_access<cl::sycl::access::mode::write>();
        for (int d = 0; d < dim; d++) {
            accessor[d] = ((double*)dat)[d];
        }
    }
    else
    if (!strcmp(name,"eps")) {
        if (eps_p == nullptr) eps_p = new cl::sycl::buffer<double,1>(cl::sycl::range<1>(dim));
        auto accessor = (*eps_p).get_access<cl::sycl::access::mode::write>();
        for (int d = 0; d < dim; d++) {
            accessor[d] = ((double*)dat)[d];
        }
    }
    else
    if (!strcmp(name,"lambda")) {
        if (lambda_p == nullptr) lambda_p = new cl::sycl::buffer<double,1>(cl::sycl::range<1>(dim));
        auto accessor = (*lambda_p).get_access<cl::sycl::access::mode::write>();
        for (int d = 0; d < dim; d++) {
            accessor[d] = ((double*)dat)[d];
        }
    }
    else
    if (!strcmp(name,"dt")) {
        if (dt_p == nullptr) dt_p = new cl::sycl::buffer<double,1>(cl::sycl::range<1>(dim));
        auto accessor = (*dt_p).get_access<cl::sycl::access::mode::write>();
        for (int d = 0; d < dim; d++) {
            accessor[d] = ((double*)dat)[d];
        }
    }
    else
    if (!strcmp(name,"del2")) {
        if (del2_p == nullptr) del2_p = new cl::sycl::buffer<double,1>(cl::sycl::range<1>(dim));
        auto accessor = (*del2_p).get_access<cl::sycl::access::mode::write>();
        for (int d = 0; d < dim; d++) {
            accessor[d] = ((double*)dat)[d];
        }
    }
    else
    if (!strcmp(name,"akap2")) {
        if (akap2_p == nullptr) akap2_p = new cl::sycl::buffer<double,1>(cl::sycl::range<1>(dim));
        auto accessor = (*akap2_p).get_access<cl::sycl::access::mode::write>();
        for (int d = 0; d < dim; d++) {
            accessor[d] = ((double*)dat)[d];
        }
    }
    else
    if (!strcmp(name,"tvdsmu")) {
        if (tvdsmu_p == nullptr) tvdsmu_p = new cl::sycl::buffer<double,1>(cl::sycl::range<1>(dim));
        auto accessor = (*tvdsmu_p).get_access<cl::sycl::access::mode::write>();
        for (int d = 0; d < dim; d++) {
            accessor[d] = ((double*)dat)[d];
        }
    }
    else
    if (!strcmp(name,"con")) {
        if (con_p == nullptr) con_p = new cl::sycl::buffer<double,1>(cl::sycl::range<1>(dim));
        auto accessor = (*con_p).get_access<cl::sycl::access::mode::write>();
        for (int d = 0; d < dim; d++) {
            accessor[d] = ((double*)dat)[d];
        }
    }
    else
    if (!strcmp(name,"Mach")) {
        if (Mach_p == nullptr) Mach_p = new cl::sycl::buffer<double,1>(cl::sycl::range<1>(dim));
        auto accessor = (*Mach_p).get_access<cl::sycl::access::mode::write>();
        for (int d = 0; d < dim; d++) {
            accessor[d] = ((double*)dat)[d];
        }
    }
    else
    if (!strcmp(name,"xt")) {
        if (xt_p == nullptr) xt_p = new cl::sycl::buffer<double,1>(cl::sycl::range<1>(dim));
        auto accessor = (*xt_p).get_access<cl::sycl::access::mode::write>();
        for (int d = 0; d < dim; d++) {
            accessor[d] = ((double*)dat)[d];
        }
    }
    else
    if (!strcmp(name,"scale")) {
        if (scale_p == nullptr) scale_p = new cl::sycl::buffer<int,1>(cl::sycl::range<1>(dim));
        auto accessor = (*scale_p).get_access<cl::sycl::access::mode::write>();
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
#include "gridgen_kernel_kernel.cpp"
#include "init_kernel_kernel.cpp"
#include "save_kernel_kernel.cpp"
#include "calvar_kernel_kernel.cpp"
#include "xder1_kernel_kernel.cpp"
#include "residue_eval_kernel.cpp"
#include "updateRK3_kernel_kernel.cpp"
#include "Riemann_kernel_kernel.cpp"
#include "limiter_kernel_kernel.cpp"
#include "tvd_kernel_kernel.cpp"
#include "vars_kernel_kernel.cpp"
#include "calupwindeff_kernel_kernel.cpp"
#include "fact_kernel_kernel.cpp"
#include "update_kernel_kernel.cpp"
#include "checkop_kernel_kernel.cpp"
