//Kernels for lattice-boltzmann demo app
#include <math.h>
#include "kernel_functions.h"


void set_zero(ACC<double> &A) {
  A(0,0,0) = 0.0;
}
