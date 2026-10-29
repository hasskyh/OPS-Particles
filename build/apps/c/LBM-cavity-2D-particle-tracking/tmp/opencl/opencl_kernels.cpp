// Auto-generated at 2026-10-29 00:14:13.394217 by ops-translator legacy

//header
#define OPS_API 2
#define OPS_2D
#include "stdlib.h"
#include "stdio.h"
#include "ops_lib_core.h"
#include "ops_opencl_rt_support.h"
#ifdef OPS_MPI
#include "ops_mpi_core.h"
#endif
// global constants
extern double OMEGA;
extern double rho0;


void ops_init_backend() {}

//this needs to be a platform specific copy symbol to device function
void ops_decl_const_char(OPS_instance *instance, int dim, char const * type, int typeSize, char * dat, char const * name ) {
  ops_execute(instance);
  cl_int ret = 0;
  if (instance->opencl_instance->OPS_opencl_core.constant == NULL) {
      instance->opencl_instance->OPS_opencl_core.constant = (cl_mem*) malloc((2)*sizeof(cl_mem));
      for (int i = 0; i < 2; i++) {
          instance->opencl_instance->OPS_opencl_core.constant[i] = NULL;
      }
  }
  if (!strcmp(name,"OMEGA")) {
      if (instance->opencl_instance->OPS_opencl_core.constant[0] == NULL) {
          instance->opencl_instance->OPS_opencl_core.constant[0] = clCreateBuffer(instance->opencl_instance->OPS_opencl_core.context, CL_MEM_READ_ONLY, dim*typeSize, NULL, &ret);
          clSafeCall( ret );
      }
      //Write the new constant to the memory of the device
      clSafeCall( clEnqueueWriteBuffer(instance->opencl_instance->OPS_opencl_core.command_queue, instance->opencl_instance->OPS_opencl_core.constant[0], CL_TRUE, 0, dim*typeSize, (void*) dat, 0, NULL, NULL) );
      clSafeCall( clFlush(instance->opencl_instance->OPS_opencl_core.command_queue) );
      clSafeCall( clFinish(instance->opencl_instance->OPS_opencl_core.command_queue) );
  }
  else
  if (!strcmp(name,"rho0")) {
      if (instance->opencl_instance->OPS_opencl_core.constant[1] == NULL) {
          instance->opencl_instance->OPS_opencl_core.constant[1] = clCreateBuffer(instance->opencl_instance->OPS_opencl_core.context, CL_MEM_READ_ONLY, dim*typeSize, NULL, &ret);
          clSafeCall( ret );
      }
      //Write the new constant to the memory of the device
      clSafeCall( clEnqueueWriteBuffer(instance->opencl_instance->OPS_opencl_core.command_queue, instance->opencl_instance->OPS_opencl_core.constant[1], CL_TRUE, 0, dim*typeSize, (void*) dat, 0, NULL, NULL) );
      clSafeCall( clFlush(instance->opencl_instance->OPS_opencl_core.command_queue) );
      clSafeCall( clFinish(instance->opencl_instance->OPS_opencl_core.command_queue) );
  }
  else
  {
    throw OPSException(OPS_RUNTIME_ERROR, "error: unknown const name");
}
}



  void buildOpenCLKernels(OPS_instance *instance) {
    static bool isbuilt = false;

    if(!isbuilt) {
      //clSafeCall( clUnloadCompiler() );

      instance->opencl_instance->OPS_opencl_core.n_kernels = 14;
      instance->opencl_instance->OPS_opencl_core.kernel = (cl_kernel*) malloc(14*sizeof(cl_kernel));
    }
    isbuilt = true;
  }
  
//user kernel files
#include "Ker_Diffuse_Refl_Norm_xpos_opencl_kernel.cpp"
#include "Ker_Diffuse_Refl_Norm_xneg_opencl_kernel.cpp"
#include "Ker_Diffuse_Refl_Norm_ypos_opencl_kernel.cpp"
#include "Ker_Diffuse_Refl_Norm_yneg_opencl_kernel.cpp"
#include "Ker_Diffuse_Refl_Norm_xneg_yneg_opencl_kernel.cpp"
#include "Ker_Diffuse_Refl_Norm_xpos_yneg_opencl_kernel.cpp"
#include "Ker_Diffuse_Refl_xpos_ypos_opencl_kernel.cpp"
#include "Ker_Diffuse_Refl_Norm_xneg_ypos_opencl_kernel.cpp"
#include "KerInitGrid_opencl_kernel.cpp"
#include "KerInitMacros_opencl_kernel.cpp"
#include "KerInitF_opencl_kernel.cpp"
#include "KerComputeMacros_opencl_kernel.cpp"
#include "KerCollision_opencl_kernel.cpp"
#include "KerStream_opencl_kernel.cpp"
