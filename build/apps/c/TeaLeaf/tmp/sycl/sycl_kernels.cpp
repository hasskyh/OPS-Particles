// Auto-generated at 2026-10-29 00:14:10.930534 by ops-translator legacy

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
cl::sycl::buffer<field_type,1> *field_p=nullptr;
extern field_type field;
cl::sycl::buffer<grid_type,1> *grid_p=nullptr;
extern grid_type grid;
cl::sycl::buffer<int,1> *number_of_states_p=nullptr;
extern int number_of_states;
cl::sycl::buffer<state_type,1> *states_p=nullptr;
extern state_type *states;
cl::sycl::buffer<int,1> *g_circ_p=nullptr;
extern int g_circ;
cl::sycl::buffer<int,1> *g_point_p=nullptr;
extern int g_point;
cl::sycl::buffer<int,1> *g_rect_p=nullptr;
extern int g_rect;

void ops_init_backend() {}

void ops_decl_const_char(OPS_instance *instance, int dim, char const * type, int size, char * dat, char const * name ) {
    if (!strcmp(name,"field")) {
        if (field_p == nullptr) field_p = new cl::sycl::buffer<field_type,1>(cl::sycl::range<1>(dim));
        auto accessor = (*field_p).get_access<cl::sycl::access::mode::write>();
        for (int d = 0; d < dim; d++) {
            accessor[d] = ((field_type*)dat)[d];
        }
    }
    else
    if (!strcmp(name,"grid")) {
        if (grid_p == nullptr) grid_p = new cl::sycl::buffer<grid_type,1>(cl::sycl::range<1>(dim));
        auto accessor = (*grid_p).get_access<cl::sycl::access::mode::write>();
        for (int d = 0; d < dim; d++) {
            accessor[d] = ((grid_type*)dat)[d];
        }
    }
    else
    if (!strcmp(name,"number_of_states")) {
        if (number_of_states_p == nullptr) number_of_states_p = new cl::sycl::buffer<int,1>(cl::sycl::range<1>(dim));
        auto accessor = (*number_of_states_p).get_access<cl::sycl::access::mode::write>();
        for (int d = 0; d < dim; d++) {
            accessor[d] = ((int*)dat)[d];
        }
    }
    else
    if (!strcmp(name,"states")) {
        if (states_p == nullptr) states_p = new cl::sycl::buffer<state_type,1>(cl::sycl::range<1>(dim));
        auto accessor = (*states_p).get_access<cl::sycl::access::mode::write>();
        for (int d = 0; d < dim; d++) {
            accessor[d] = ((state_type*)dat)[d];
        }
    }
    else
    if (!strcmp(name,"g_circ")) {
        if (g_circ_p == nullptr) g_circ_p = new cl::sycl::buffer<int,1>(cl::sycl::range<1>(dim));
        auto accessor = (*g_circ_p).get_access<cl::sycl::access::mode::write>();
        for (int d = 0; d < dim; d++) {
            accessor[d] = ((int*)dat)[d];
        }
    }
    else
    if (!strcmp(name,"g_point")) {
        if (g_point_p == nullptr) g_point_p = new cl::sycl::buffer<int,1>(cl::sycl::range<1>(dim));
        auto accessor = (*g_point_p).get_access<cl::sycl::access::mode::write>();
        for (int d = 0; d < dim; d++) {
            accessor[d] = ((int*)dat)[d];
        }
    }
    else
    if (!strcmp(name,"g_rect")) {
        if (g_rect_p == nullptr) g_rect_p = new cl::sycl::buffer<int,1>(cl::sycl::range<1>(dim));
        auto accessor = (*g_rect_p).get_access<cl::sycl::access::mode::write>();
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
#include "field_summary_kernel_kernel.cpp"
#include "generate_chunk_kernel_kernel.cpp"
#include "initialise_chunk_kernel_zero_kernel.cpp"
#include "initialise_chunk_kernel_zero_x_kernel.cpp"
#include "initialise_chunk_kernel_zero_y_kernel.cpp"
#include "initialise_chunk_kernel_xx_kernel.cpp"
#include "initialise_chunk_kernel_yy_kernel.cpp"
#include "initialise_chunk_kernel_x_kernel.cpp"
#include "initialise_chunk_kernel_y_kernel.cpp"
#include "initialise_chunk_kernel_cellx_kernel.cpp"
#include "initialise_chunk_kernel_celly_kernel.cpp"
#include "initialise_chunk_kernel_volume_kernel.cpp"
#include "set_field_kernel_kernel.cpp"
#include "tea_leaf_init_zero2_kernel_kernel.cpp"
#include "tea_leaf_yeqx_kernel_kernel.cpp"
#include "tea_leaf_dot_kernel_kernel.cpp"
#include "tea_leaf_cg_calc_w_reduce_kernel_kernel.cpp"
#include "tea_leaf_axpy_kernel_kernel.cpp"
#include "tea_leaf_cg_calc_ur_r_reduce_kernel_kernel.cpp"
#include "tea_leaf_axpby_kernel_kernel.cpp"
#include "tea_leaf_cheby_init_kernel_kernel.cpp"
#include "tea_leaf_recip3_kernel_kernel.cpp"
#include "tea_leaf_xpy_kernel_kernel.cpp"
#include "tea_leaf_common_init_u_u0_kernel_kernel.cpp"
#include "tea_leaf_recip_kernel_kernel.cpp"
#include "tea_leaf_common_init_Kx_Ky_kernel_kernel.cpp"
#include "tea_leaf_init_zero_kernel_kernel.cpp"
#include "tea_leaf_common_init_kernel_kernel.cpp"
#include "tea_leaf_recip2_kernel_kernel.cpp"
#include "tea_leaf_common_residual_kernel_kernel.cpp"
#include "tea_leaf_norm2_kernel_kernel.cpp"
#include "tea_leaf_common_init_diag_init_kernel_kernel.cpp"
#include "tea_leaf_zeqxty_kernel_kernel.cpp"
#include "tea_leaf_jacobi_kernel_kernel.cpp"
#include "tea_leaf_ppcg_init1_kernel_kernel.cpp"
#include "tea_leaf_ppcg_init2_kernel_kernel.cpp"
#include "tea_leaf_ppcg_inner1_kernel_kernel.cpp"
#include "tea_leaf_ppcg_inner2_kernel_kernel.cpp"
#include "tea_leaf_ppcg_reduce_kernel_kernel.cpp"
#include "update_halo_kernel1_b2_kernel.cpp"
#include "update_halo_kernel1_b1_kernel.cpp"
#include "update_halo_kernel1_t2_kernel.cpp"
#include "update_halo_kernel1_t1_kernel.cpp"
#include "update_halo_kernel1_l2_kernel.cpp"
#include "update_halo_kernel1_l1_kernel.cpp"
#include "update_halo_kernel1_r2_kernel.cpp"
#include "update_halo_kernel1_r1_kernel.cpp"
