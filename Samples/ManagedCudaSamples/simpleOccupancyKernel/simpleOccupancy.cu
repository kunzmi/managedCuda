
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

////////////////////////////////////////////////////////////////////////////////
// Test kernel
//
// This kernel squares each array element. Each thread addresses
// himself with threadIdx and blockIdx, so that it can handle any
// execution configuration, including anything the launch configurator
// API suggests.
////////////////////////////////////////////////////////////////////////////////

extern "C"
__global__ void square(int *array, int arrayCount)
{
	extern __shared__ int dynamicSmem[];
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx < arrayCount) {
		array[idx] *= array[idx];
	}
}